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
    ap.add_argument("--max-days", type=int, default=0,
                    help=">0 = smoke: truncate training days per scenario")
    args = ap.parse_args()

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("disk:\n%s", subprocess.run(["df", "-h", "/tmp", "/kaggle/working"],
                                         capture_output=True, text=True).stdout)

    from_cache_dirs = [d for d in os.environ.get("CLIMX_CACHE_INPUT_DIRS", "").split(":") if d]
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
        stage("copy cache datasets to /tmp",
              lambda: copy_datasets_to_cache(from_cache_dirs, CACHE_DIR))
        X_test = xr.open_dataset(CACHE_DIR / "X_test.nc",
                                 chunks={"time": 512, "lat": -1, "lon": -1})
        metadata_test = _metadata_for()
        hist = open_hist_from_cache(CACHE_DIR)
        ds = None
    else:
        # ---------- build-in-session mode ----------
        full_dir = Path(os.environ.get("CLIMX_FULL_DIR", str(find_full_dir())))
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
    set_seed(42)
    emulator = UNetEmulator(dict(
        n_in=22, n_out=7, base_channels=args.base_channels,
        learning_rate=args.learning_rate,
        variable_weights=[1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
        pad_to=[192, 288], skip_connections=bool(args.skip_connections)))
    stage("fit (cache, 192x288)", lambda: emulator.fit(
        CACHE_DIR, None, trainer_params=dict(
            max_epochs=args.epochs, batch_size=args.batch, num_workers=2,
            sample_step=1, scenarios=SCENARIOS, max_days=args.max_days,
            accelerator="auto", devices=1,
            model_path="/tmp/full_unet_models",
            logger=False)))
    emulator.save("/tmp/full_unet.pt")

    # ---------- evaluate (two passes; RAM-safe) ----------
    PRED_PATH = Path("/tmp/full_unet_predictions.zarr")
    y_phys, _ = stage(
        "evaluate pass 1 (predict + reversal + save)", lambda: emulator.evaluate(
            X_test, metadata_test,
            results_path=Path("/tmp/full_unet_evaluation.json"),
            predictions_path=PRED_PATH,
            historical_data_path=hist, TARGET_VARIABLES=paths.TARGET_VARIABLES,
            LOAD_PREDICTIONS=False, compute_indices=False))
    del y_phys
    import gc
    gc.collect()
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
