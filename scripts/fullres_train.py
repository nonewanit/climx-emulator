"""Full-res U-Net TRAINING — Kaggle GPU session.

TWO MODES:
  1. Default (build-in-session): attach the raw input dataset
     `dtdoww/climx-full-resolution-data` and the whole cache is built in
     /tmp first (~2-5 h on the slow input mount).
  2. --from-cache (recommended): attach the PRIVATE cache datasets built by
     scripts/full_cache.py (CPU session) and set CLIMX_CACHE_INPUT_DIRS to
     their mount dirs (colon-separated). The files are plain-copied to /tmp
     (~30-40 min at the mount's sequential speed) and no slow reads or
     re-preprocessing happen. The raw 171 GB input is NOT needed.

Run:  %run scripts/fullres_train.py [--subsample 3] [--epochs 2] [--batch 16]
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import xarray as xr

import climx_project  # noqa: F401  (vendor + env + compat patches)
from climx_project import paths, submission
from climx_project.cachebuild import (
    copy_datasets_to_cache, open_hist_from_cache, write_cache_x, write_cache_y)
from climx_project.cacheio import zarr_fp16_encoding
from climx_project.data import daily_forcings, open_full_datasets_local
from climx_project.models.unet import UNetEmulator
from climx_project.runners.base import set_seed
from src.data_preprocessing.preprocessing import preprocess_train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fullres_train")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# NOTE: paths.PREPROCESSING_DIR was resolved at import time — the small
# artifacts (scaling JSON, mode marker) land in the repo's data/ dir inside
# /kaggle/working, which persists across sessions of the notebook.
CACHE_DIR = Path(os.environ.get("CLIMX_CACHE_DIR", "/tmp/full_cache"))
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
CLIMATOLOGY = paths.PREPROCESSING_DIR / "climatology_full.nc"
MODE_MARKER = paths.PREPROCESSING_DIR / "stationarization_mode.txt"
BUNDLED_SCALING = paths.PROJECT_ROOT / "assets" / "fullres_scaling_parameters.json"
SCENARIOS = ["ssp126", "ssp370", "ssp585"]

_CHUNKS = {"time": 365, "time_month": 24, "time_year": -1, "lat": -1, "lon": -1}
_HIST_CHUNKS = {"time": 3650, "time_month": 24, "time_year": -1, "lat": -1, "lon": -1}


def stage(name, fn):
    t0 = time.time()
    log.info("=== %s ===", name)
    try:
        out = fn()
    except Exception as e:
        log.error("FAILED %s: %r", name, e)
        raise
    log.info("OK %s (%.1f min)", name, (time.time() - t0) / 60)
    return out


def find_full_dir() -> Path:
    import pathlib
    default = Path("/kaggle/input/datasets/dtdoww/climx-full-resolution-data")
    if default.is_dir():
        return default
    hits = list(pathlib.Path("/kaggle/input").rglob(
        "NorESM2-MM_r1i1p1f1_with_forcing_train_ssp126.nc"))
    if not hits:
        raise FileNotFoundError(
            "full-res files not found under /kaggle/input — attach the dataset "
            "dtdoww/climx-full-resolution-data as a notebook input.")
    return hits[0].parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--skip-connections", type=int, default=1, choices=(0, 1))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS),
                    help="comma-separated scenarios to train on (the session "
                         "storage quota ~57.6 GiB can't hold the full x1 cache "
                         "- one scenario ~24 GB + hist ~11 GB fits)")
    ap.add_argument("--load-model", default=None,
                    help="path to a saved full_unet.pt — skip the fit and "
                         "evaluate with this model")
    ap.add_argument("--max-days", type=int, default=0,
                    help=">0 = smoke: truncate training days per scenario")
    args = ap.parse_args()

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("disk:\n%s", subprocess.run(["df", "-h", "/tmp", "/kaggle/working"],
                                         capture_output=True, text=True).stdout)

    from_cache_dirs = [d for d in os.environ.get("CLIMX_CACHE_INPUT_DIRS", "").split(":") if d]
    train_scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    mode = "none"

    def _metadata_for():
        return {
            "climatology_path": str(CLIMATOLOGY),
            "scaling_params_path": str(SCALING_JSON),
            "scaling_params": json.load(open(SCALING_JSON)),
            "stationarization_mode": mode,
        }

    if from_cache_dirs:
        # ---------- from-cache mode: copy attached datasets, no slow reads ----------
        log.info("from-cache mode: %s", from_cache_dirs)
        if not SCALING_JSON.exists() and BUNDLED_SCALING.exists():
            SCALING_JSON.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BUNDLED_SCALING, SCALING_JSON)
            MODE_MARKER.write_text("none")
        # The evaluate needs: the raw test forcings (1.4 GB, via HF) and the
        # hist window (read from the mount on demand). X_test.nc and the
        # hist packages are NOT copied: the 86-year submission makes the
        # preprocessed X_test ~152 GB, so the forcings are transformed
        # lazily per predict chunk instead.
        if args.load_model:
            stage("copy cache datasets to /tmp",
                  lambda: copy_datasets_to_cache(from_cache_dirs, CACHE_DIR,
                                                 only_scenarios=[],
                                                 skip_x_test=True, skip_hist=True))
        else:
            stage("copy cache datasets to /tmp",
                  lambda: copy_datasets_to_cache(from_cache_dirs, CACHE_DIR,
                                                 only_scenarios=train_scenarios,
                                                 skip_x_test=True, skip_hist=True))
        metadata_test = _metadata_for()
        hist = open_hist_from_cache(CACHE_DIR)
        ds = None
    else:
        # ---------- build-in-session mode ----------
        # Download the raw files via HF (measured ~62 MB/s on Kaggle) into
        # /tmp so every read after is local disk. This is the ONE-SHOT Phase B
        # path: one notebook, one 9 h session, no datasets, no uploads.
        mount_dir = Path(os.environ.get("CLIMX_FULL_DIR", str(find_full_dir())))
        RAW_TMP = Path("/tmp/raw_full")
        RAW_TMP.mkdir(parents=True, exist_ok=True)
        PREFIX = "NorESM2-MM_r1i1p1f1_with_forcing_"
        for n in ("hist_train_1850_1931", "hist_train_1932_2014",
                  "train_ssp126", "train_ssp370", "train_ssp585",
                  "test_forcings_ssp245"):
            dst = RAW_TMP / f"{PREFIX}{n}.nc"
            if dst.exists() and dst.stat().st_size > 1e9:
                continue
            try:
                from huggingface_hub import hf_hub_download
                log.info("hf download %s", f"{PREFIX}{n}.nc")
                p = hf_hub_download(
                    repo_id="isp-uv-es/ClimX", repo_type="dataset",
                    filename=f"full/{PREFIX}{n}.nc",
                    local_dir=str(Path("/tmp/hf_tmp")))
                shutil.move(p, dst)
            except Exception as e:
                log.warning("hf download failed (%s) — mount copy fallback", e)
                shutil.copy2(mount_dir / f"{PREFIX}{n}.nc", dst)
        full_dir = RAW_TMP
        log.info("full-res data dir: %s", full_dir)
        ds = stage("open local full-res datasets", lambda: open_full_datasets_local(
            full_dir, scenarios=SCENARIOS))
        log.info("hist: %s | train: %s", dict(ds.hist.sizes), dict(ds.train.sizes))
        ds.hist = ds.hist.chunk(_HIST_CHUNKS)
        ds.train = ds.train.chunk(_CHUNKS)
        ds.hist = daily_forcings(ds.hist)

        if not SCALING_JSON.exists() and BUNDLED_SCALING.exists():
            log.info("using repo-bundled scaling params (%s)", BUNDLED_SCALING)
            SCALING_JSON.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BUNDLED_SCALING, SCALING_JSON)
            MODE_MARKER.write_text("none")

        ds.test_forcings = ds.test_forcings.sel(time=slice("2081-01-01", "2101-01-01"))
        X_TEST_NC = CACHE_DIR / "X_test.nc"
        if X_TEST_NC.exists():
            try:
                X_test = xr.open_dataset(X_TEST_NC, chunks={"time": 512, "lat": -1, "lon": -1})
                _ = dict(X_test.sizes)
                metadata_test = _metadata_for()
            except Exception:
                log.warning("X_test.nc is corrupt — removing and redoing")
                X_TEST_NC.unlink()
        if not X_TEST_NC.exists():
            if mode == "none" and not CLIMATOLOGY.exists():
                xr.Dataset().to_netcdf(CLIMATOLOGY)
            from climx_project.preprocess import _save_chunked
            from src.data_preprocessing.preprocessing import preprocess_test
            X_test, metadata_test = stage(
                "preprocess_test (SSP2-4.5 forcings)", lambda: preprocess_test(
                    ds, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
                    scaling_params_path=SCALING_JSON, load_into_memory=False,
                    stationarization_mode=mode))
            metadata_test["stationarization_mode"] = mode
            X_test = X_test.astype("float32")  # netCDF4 has no float16; f4 still halves the float64 bloat
            encoding = {v: {"zlib": False, "_FillValue": None} for v in X_test.data_vars}
            _save_chunked(X_test, X_TEST_NC, encoding=encoding)
        log.info("X_test: %s", dict(X_test.sizes))

        ds.train = ds.train.isel(time=slice(0, None, args.subsample))
        X_scaled, _, _ = stage("pass B: lazy scaled fields", lambda: preprocess_train(
            ds, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
            scaling_params_path=SCALING_JSON, load_into_memory=False,
            stationarization_mode=mode))

        ENCODING = zarr_fp16_encoding()
        write_cache_x(X_scaled, CACHE_DIR, SCENARIOS, ENCODING)
        write_cache_y(full_dir, CACHE_DIR, SCENARIOS, ENCODING, SCALING_JSON)
        hist = ds.hist

    # ---------- fit (shared) ----------
    if args.load_model:
        emulator = UNetEmulator(dict(
            n_in=22, n_out=7, base_channels=args.base_channels,
            learning_rate=args.learning_rate,
            variable_weights=[1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
            pad_to=[192, 288], skip_connections=bool(args.skip_connections)))
        stage("load model", lambda: emulator.load(args.load_model))
        log.info("fit SKIPPED — evaluating %s", args.load_model)
    else:
        set_seed(42)
        emulator = UNetEmulator(dict(
            n_in=22, n_out=7, base_channels=args.base_channels,
            learning_rate=args.learning_rate,
            variable_weights=[1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
            pad_to=[192, 288], skip_connections=bool(args.skip_connections)))
        stage("fit (cache, 192x288)", lambda: emulator.fit(
            CACHE_DIR, None, trainer_params=dict(
                max_epochs=args.epochs, batch_size=args.batch, num_workers=2,
                sample_step=1, scenarios=train_scenarios, max_days=args.max_days,
                accelerator="auto", devices=1,
                model_path="/tmp/full_unet_models",
                logger=False)))
        emulator.save("/tmp/full_unet.pt")
        # The scenario stores are dead weight after training — free the
        # storage quota before the evaluate.
        for dead in list(CACHE_DIR.glob("X_*.zarr")) + list(CACHE_DIR.glob("y_*.zarr")):
            shutil.rmtree(dead)
    # Free the training-side + GPU memory before the evaluate: predict
    # accumulates ~12 GB of predictions and the reversal copies it; the GPU
    # training arena otherwise lingers and the forward OOMs.
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- evaluate (86-year submission) ----------
    # The submission wants the indices over the FULL 2015-2100 span
    # (86 yearly values x grid — verified against sample_submission). The
    # preprocessed X_test for that span would be ~152 GB, so the test
    # forcings are fetched (1.4 GB via HF) and transformed lazily per
    # predict chunk.
    from climx_project.fullres_eval import (evaluate_pass1,
                                            open_hist_from_mount)
    raw_test = Path("/tmp/raw_full") / "NorESM2-MM_r1i1p1f1_with_forcing_test_forcings_ssp245.nc"
    if not raw_test.exists():
        from huggingface_hub import hf_hub_download
        log.info("hf download test forcings (1.4 GB)")
        p_dl = hf_hub_download(
            repo_id="isp-uv-es/ClimX", repo_type="dataset",
            filename="full/NorESM2-MM_r1i1p1f1_with_forcing_test_forcings_ssp245.nc",
            local_dir=str(Path("/tmp/hf_tmp")))
        raw_test.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(p_dl, raw_test)
    PRED_PATH = Path("/tmp/full_unet_predictions.zarr")
    stage("evaluate pass 1 (predict + reversal + save, fp16)",
          lambda: evaluate_pass1(emulator, raw_test, metadata_test, PRED_PATH))
    X_test = raw_test  # pass 2 loads the saved predictions; X_test is unused
    hist = stage("open hist window from mount",
                 lambda: open_hist_from_mount(from_cache_dirs, CACHE_DIR))
    _, indices_pred = stage(
        "evaluate pass 2 (indices from saved predictions)", lambda: emulator.evaluate(
            X_test, metadata_test,
            results_path=Path("/tmp/full_unet_evaluation.json"),
            predictions_path=PRED_PATH,
            historical_data_path=hist, TARGET_VARIABLES=paths.TARGET_VARIABLES,
            LOAD_PREDICTIONS=True, compute_indices=True))
    parquet = stage("build submission parquet", lambda: submission.build_submission(
        indices_pred, "unet_full"))
    report = submission.verify_parquet(parquet)
    log.info("parquet verified: %s", report)
    log.info("FULLRES_TRAIN_OK — %s", parquet)


if __name__ == "__main__":
    main()
