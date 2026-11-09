"""Full-res U-Net TRAINING — Kaggle interactive notebook, LOCAL data.

Attach the dataset `dtdoww/climx-full-resolution-data` as a notebook input;
everything is read from the local mount (no HF streaming, ~1 GB/s reads).

Per-session pipeline (each stage is idempotent where possible):
  1. open the local full-res files (hist + train x3 scenarios + SSP2-4.5 test)
  2. climatology + scaling params — computed ONCE, persisted in /kaggle/working
  3. preprocess_test -> X_test.nc — ONCE, persisted (~3.4 GB, fits the 20 GB)
  4. write the subsampled fp16 training cache to /tmp (dies with the session,
     ~10-20 min — the local mount makes this cheap)
  5. fit the U-Net from the cache (GPU, devices=1)
  6. predict + full index computation + Kaggle submission parquet

Run:  %run scripts/fullres_train.py [--subsample 3] [--epochs 10] [--batch 16]
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

import climx_project  # noqa: F401  (vendor + env + compat patches)
from climx_project import paths, submission
from climx_project.data import daily_forcings, open_full_datasets_local
from climx_project.models.unet import UNetEmulator
from climx_project.runners.base import set_seed
from src.data_preprocessing.preprocessing import preprocess_train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fullres_train")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

os.environ.setdefault("CLIMX_PREPROCESS_DIR", "/kaggle/working/data/preprocessing_data_full")
CACHE_DIR = Path(os.environ.get("CLIMX_CACHE_DIR", "/tmp/full_cache"))
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
CLIMATOLOGY = paths.PREPROCESSING_DIR / "climatology_full.nc"
SCENARIOS = ["ssp126", "ssp370", "ssp585"]

# Native-res forcing dims must be chunked BEFORE the daily upsample expands
# them (one full-var chunk would expand to ~60k days in a single dask task).
_CHUNKS = {"time": 365, "time_month": 24, "time_year": -1, "lat": -1, "lon": -1}


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
    default = Path("/kaggle/input/datasets/dtdoww/climx-full-resolution-data")
    if default.is_dir():
        return default
    # mount layouts vary; search any input mount for the file prefix
    import pathlib
    hits = list(pathlib.Path("/kaggle/input").rglob(
        "NorESM2-MM_r1i1p1f1_with_forcing_train_ssp126.nc"))
    if not hits:
        raise FileNotFoundError(
            "full-res files not found under /kaggle/input — attach the dataset "
            "dtdoww/climx-full-resolution-data as a notebook input (Data -> Add Input).")
    return hits[0].parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=3,
                    help="write every Nth training day to the /tmp cache")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--skip-connections", type=int, default=1, choices=(0, 1))
    ap.add_argument("--max-days", type=int, default=0,
                    help=">0 = smoke: truncate training days per scenario")
    args = ap.parse_args()

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full_dir = Path(os.environ.get("CLIMX_FULL_DIR", str(find_full_dir())))
    log.info("full-res data dir: %s", full_dir)

    # 1. Open everything locally (metadata only, seconds).
    ds = stage("open local full-res datasets", lambda: open_full_datasets_local(
        full_dir, scenarios=SCENARIOS))
    log.info("hist: %s | train: %s | test: %s",
             dict(ds.hist.sizes), dict(ds.train.sizes), dict(ds.test_forcings.sizes))

    ds.hist = ds.hist.chunk(_CHUNKS)
    ds.train = ds.train.chunk(_CHUNKS)
    ds.hist = daily_forcings(ds.hist)

    # 2. Climatology + scaling params (once; persisted in /kaggle/working).
    if CLIMATOLOGY.exists() and SCALING_JSON.exists():
        log.info("climatology + scaling params exist — skipping pass A")
    else:
        hist_s = ds.hist.isel(time=slice(0, None, 3))
        tiny = type("DS", (), {
            "hist": hist_s,
            "train": ds.train.isel(time=slice(0, 2000)),
            "test_forcings": ds.test_forcings,
        })()
        with ProgressBar():
            stage("pass A: climatology + scaling params (local hist + sample)",
                  lambda: preprocess_train(
                      tiny, preprocessing_path=paths.PREPROCESSING_DIR,
                      version="full", scaling_params_path=None,
                      load_into_memory=False, stationarization_mode="monthly"))

    # 3. Test preprocessing (once; X_test.nc persists, ~3.4 GB).
    if not paths.X_TEST_NC.exists():
        from climx_project.preprocess import run_preprocess_test
        X_test, metadata_test = stage(
            "preprocess_test (SSP2-4.5 forcings)", lambda: run_preprocess_test(
                ds, version="full", load_into_memory=False))
    else:
        from climx_project.preprocess import load_metadata
        X_test = xr.open_dataset(paths.X_TEST_NC, chunks={"time": 512, "lat": -1, "lon": -1})
        metadata_test = load_metadata()
    log.info("X_test: %s", dict(X_test.sizes))

    # 4. Write the subsampled fp16 training cache to /tmp (per session).
    ds.train = ds.train.isel(time=slice(0, None, args.subsample))
    X_scaled, y_scaled, _ = stage(
        "pass B: lazy scaled anomalies (subsampled train)",
        lambda: preprocess_train(
            ds, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
            scaling_params_path=SCALING_JSON, load_into_memory=False,
            stationarization_mode="monthly"))

    from numcodecs import Blosc
    ENCODING = {"dtype": "float16",
                "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)}

    def write_cache(dsx, prefix):
        plan = [(s, n) for s in SCENARIOS for n in dsx.data_vars]
        total_gb = sum(float(dsx[n].size) * 2 / 1e9 / len(SCENARIOS) for _, n in plan)
        log.info("cache vars (%s): %d — %.1f GB", prefix, len(plan), total_gb)
        done_gb = 0.0
        rate = None
        for i, (scen, name) in enumerate(plan):
            key = f"{prefix}_{name}_{scen}"
            out = CACHE_DIR / f"{key}.zarr"
            if out.exists():
                continue
            tmp = CACHE_DIR / f".tmp_{key}.zarr"
            if tmp.exists():
                shutil.rmtree(tmp)
            gb = float(dsx[name].size) * 2 / 1e9 / len(SCENARIOS)
            log.info("=== write %s (%d/%d, %.1f GB) ===", key, i + 1, len(plan), gb)
            t0 = time.time()
            with ProgressBar():
                dsx.sel(forcing_scenario=scen)[[name]].astype("float16").to_zarr(
                    tmp, mode="w", encoding={name: ENCODING})
            tmp.rename(out)
            mins = (time.time() - t0) / 60
            rate = gb / mins
            done_gb += gb
            eta_h = (total_gb - done_gb) / rate / 60 if rate else float("nan")
            log.info("OK %s: %.1f min, %.2f GB/min | done %.1f/%.1f GB | ETA %.1f h",
                     key, mins, rate, done_gb, total_gb, eta_h)

    write_cache(X_scaled, "X")
    write_cache(y_scaled, "y")

    # 5. Fit from the cache (GPU, 1 device — DDP hangs on the 2-GPU box).
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
            model_path=str(paths.MODELS_DIR / "full_unet"),
            logger=False)))
    emulator.save(paths.MODELS_DIR / "full_unet.pt")

    # 6. Evaluate: predict + reversal + FULL index computation + submission.
    y_phys, indices_pred = stage(
        "evaluate (indices ON, SSP2-4.5 2081-2100)", lambda: emulator.evaluate(
            X_test, metadata_test,
            results_path=paths.INDICES_DIR / "full_unet_evaluation.json",
            predictions_path=paths.PREDICTIONS_DIR / "full_unet_predictions.zarr",
            historical_data_path=ds.hist, TARGET_VARIABLES=paths.TARGET_VARIABLES,
            LOAD_PREDICTIONS=False, compute_indices=True))
    parquet = stage("build submission parquet", lambda: submission.build_submission(
        indices_pred, "unet_full"))
    report = submission.verify_parquet(parquet)
    log.info("parquet verified: %s", report)
    log.info("FULLRES_TRAIN_OK — %s", parquet)


if __name__ == "__main__":
    main()
