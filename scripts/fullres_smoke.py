"""Full-resolution U-Net SMOKE TEST — runs on Kaggle's GPU (interactive notebook).

Proves the full-grid code path without downloading the 170 GB dataset:
- opens the full-res NetCDFs REMOTELY (HF filesystem, range requests),
  sliced to a few hundred days,
- runs the real preprocessing on the full 192x288 grid (climatology computed
  from the train slice — wrong values, valid plumbing),
- trains the U-Net 1 epoch (pad_to [192,288], skip connections ON, 1 device),
- predicts on a small test slice and prints the shapes.

IDEMPOTENT: if the preprocessed X/y .nc files exist from a previous run (they
persist in /kaggle/working between notebook sessions), stages 1-3 are skipped
and the arrays are loaded from disk — reruns cost minutes, not hours.

Every stage prints OK/FAIL; the final line is "FULLRES_SMOKE_OK".
Run it interactively with `%run scripts/fullres_smoke.py` for live output.
"""

import json
import logging
import os
import sys
import time

import xarray as xr

import climx_project  # noqa: F401  (vendor + env + compat patches)
from climx_project import preprocess, paths
from climx_project.data import daily_forcings, open_full_datasets
from climx_project.models.unet import UNetEmulator
from climx_project.runners.base import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fullres_smoke")
# The remote reads log every range request at INFO — thousands of URL lines
# drown the stage markers. Keep the log readable in the notebook console.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

os.environ.setdefault("CLIMX_PREPROCESS_DIR", "/kaggle/working/data/preprocessing_data_full_smoke")
paths.ensure_dirs()  # results/{models,predictions,indices} for the later stages
PREPROC_DIR = paths.PREPROCESSING_DIR
PREPROC_DIR.mkdir(parents=True, exist_ok=True)
SCALING_FULL = PREPROC_DIR / "scaling_parameters_full.json"
CLIMATOLOGY_FULL = PREPROC_DIR / "climatology_full.nc"


def stage(name, fn):
    t0 = time.time()
    log.info("=== %s ===", name)
    try:
        out = fn()
    except Exception as e:
        log.error("FAILED %s: %r", name, e)
        raise
    log.info("OK %s (%.1f s)", name, time.time() - t0)
    return out


# 1. Remote open (slices: 400 days x 2 train scenarios, 30 test days).
# 400 days >= one full year: monthly_climatology_to_daily needs months 1..12.
# Cheap (metadata only, ~10 s) — always run, even when preprocessing is cached.
ds = stage("open full-res NetCDFs remotely", lambda: open_full_datasets(
    train_days=400, test_days=30))
log.info("train dims: %s", dict(ds.train.sizes))
log.info("test dims: %s", dict(ds.test_forcings.sizes))

preprocessed = all(p.exists() for p in (paths.X_TRAIN_NC, paths.Y_TRAIN_NC,
                                        paths.X_TEST_NC)) and SCALING_FULL.exists()

if preprocessed:
    log.info("preprocessed smoke data found on disk — skipping preprocessing")
    X_train = xr.open_dataset(paths.X_TRAIN_NC, chunks={"time": 100, "lat": -1, "lon": -1})
    y_train = xr.open_dataset(paths.Y_TRAIN_NC, chunks={"time": 100, "lat": -1, "lon": -1})
    X_test = xr.open_dataset(paths.X_TEST_NC, chunks={"time": 100, "lat": -1, "lon": -1})
    with open(SCALING_FULL) as f:
        scaling_params = json.load(f)
    metadata_test = {
        "climatology_path": str(CLIMATOLOGY_FULL),
        "scaling_params_path": str(SCALING_FULL),
        "scaling_params": scaling_params,
        "stationarization_mode": "monthly",
    }
else:
    # 2. The FULL files store forcings at native monthly/yearly resolution; the
    # starter's preprocessing expects a daily time grid (lite had it pre-merged).
    stage("upsample forcings to daily", lambda: None)  # marker only
    ds.train = daily_forcings(ds.train)
    log.info("after daily_forcings, train dims: %s", dict(ds.train.sizes))

    # 3. Real preprocessing on the full grid (the slow stage: ~2-3 h streaming).
    X_train, y_train, _metadata = stage(
        "preprocess_train (full grid, smoke slice)",
        lambda: preprocess.run_preprocess_train(
            ds, version="full", load_into_memory=False))
    X_test, metadata_test = stage(
        "preprocess_test (full grid, smoke slice)",
        lambda: preprocess.run_preprocess_test(ds, version="full", load_into_memory=False))
    log.info("X_train: %s | X_test: %s", dict(X_train.sizes), dict(X_test.sizes))

# 4. U-Net fit, 1 epoch, full grid, skips ON. devices=1 is REQUIRED: with
# accelerator="auto" on a 2-GPU box Lightning launches DDP and hangs waiting
# for the second client (v5 failure: DistStoreError timeout after 1801 s).
set_seed(42)
emulator = UNetEmulator(dict(
    n_in=22, n_out=7, base_channels=64, learning_rate=1e-3,
    variable_weights=[1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
    pad_to=[192, 288], skip_connections=True))
stage("fit 1 epoch (192x288, skips ON, 1 GPU)", lambda: emulator.fit(
    X_train, y_train, trainer_params=dict(
        max_epochs=1, batch_size=4, num_workers=2, max_days=60,
        accelerator="auto", devices=1,
        model_path=str(paths.MODELS_DIR / "fullres_smoke"),
        logger=False)))

# 5. Predict on the 30-day test slice.
y_pred = stage("predict 30-day test slice", lambda: emulator.predict(
    X_test, target_vars=paths.TARGET_VARIABLES))
log.info("prediction: %s", {v: tuple(y_pred[v].shape) for v in list(y_pred.data_vars)[:2]})

# 6. Preprocessing reversal end to end (indices OFF: the 30-day smoke slice
# cannot satisfy the percentile indices' 1961-1990 base period anyway — the
# reversal is what this stage proves).
y_phys, _ = stage("evaluate tail (preprocessing reversal)", lambda: emulator.evaluate(
    X_test, metadata_test,
    results_path=paths.INDICES_DIR / "fullres_smoke.json",
    predictions_path=paths.PREDICTIONS_DIR / "fullres_smoke.zarr",
    historical_data_path=ds.hist, TARGET_VARIABLES=paths.TARGET_VARIABLES,
    LOAD_PREDICTIONS=False, compute_indices=False))
log.info("reversed fields: %s", {v: tuple(y_phys[v].shape) for v in list(y_phys.data_vars)[:2]})

log.info("FULLRES_SMOKE_OK")
