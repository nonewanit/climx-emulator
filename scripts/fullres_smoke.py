"""Full-resolution U-Net SMOKE TEST — runs on Kaggle's GPU.

Proves the full-grid code path without downloading the 170 GB dataset:
- opens the full-res NetCDFs REMOTELY (HF filesystem, range requests),
  sliced to a few hundred days,
- runs the real preprocessing on the full 192x288 grid (climatology computed
  from the train slice — wrong values, valid plumbing),
- trains the U-Net 1 epoch (pad_to [192,288], skip connections ON),
- predicts on a small test slice and prints the shapes.

Every stage prints OK/FAIL; the final line is "FULLRES_SMOKE_OK".
"""

import logging
import os
import sys
import time

import climx_project  # noqa: F401  (vendor + env + compat patches)
from climx_project import preprocess, paths
from climx_project.data import open_full_datasets
from climx_project.models.unet import UNetEmulator
from climx_project.runners.base import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fullres_smoke")

os.environ.setdefault("CLIMX_PREPROCESS_DIR", "/kaggle/working/data/preprocessing_data_full_smoke")
PREPROC_DIR = paths.PREPROCESSING_DIR
PREPROC_DIR.mkdir(parents=True, exist_ok=True)


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


# 1. Remote open (slices: 120 days x 2 train scenarios, 30 test days).
ds = stage("open full-res NetCDFs remotely", lambda: open_full_datasets(
    train_days=120, test_days=30))
log.info("train dims: %s", dict(ds.train.sizes))
log.info("test dims: %s", dict(ds.test_forcings.sizes))

# 2. Smoke climatology from the train slice (avoids reading the 63 GB hist files).
ds.hist = ds.train  # same variable layout; values are wrong, plumbing is real

# 3. Real preprocessing on the full grid.
X_train, y_train, metadata = stage(
    "preprocess_train (full grid, smoke slice)",
    lambda: preprocess.run_preprocess_train(
        ds, version="full", load_into_memory=False))
X_test, metadata_test = stage(
    "preprocess_test (full grid, smoke slice)",
    lambda: preprocess.run_preprocess_test(ds, version="full", load_into_memory=False))
log.info("X_train: %s | X_test: %s", dict(X_train.sizes), dict(X_test.sizes))

# 4. U-Net fit, 1 epoch, full grid, skips ON.
set_seed(42)
emulator = UNetEmulator(dict(
    n_in=22, n_out=7, base_channels=64, learning_rate=1e-3,
    variable_weights=[1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
    pad_to=[192, 288], skip_connections=True))
stage("fit 1 epoch (192x288, skips ON)", lambda: emulator.fit(
    X_train, y_train, trainer_params=dict(
        max_epochs=1, batch_size=4, num_workers=2, max_days=60,
        accelerator="auto", model_path=str(paths.MODELS_DIR / "fullres_smoke"),
        logger=False)))

# 5. Predict on the 30-day test slice.
y_pred = stage("predict 30-day test slice", lambda: emulator.predict(
    X_test, target_vars=paths.TARGET_VARIABLES))
log.info("prediction: %s", {v: tuple(y_pred[v].shape) for v in list(y_pred.data_vars)[:2]})

# 6. Preprocessing reversal + index computation, end to end.
stage("evaluate tail (reversal + indices)", lambda: emulator.evaluate(
    X_test, metadata_test,
    results_path=paths.INDICES_DIR / "fullres_smoke.json",
    predictions_path=paths.PREDICTIONS_DIR / "fullres_smoke.zarr",
    historical_data_path=ds.hist, TARGET_VARIABLES=paths.TARGET_VARIABLES,
    LOAD_PREDICTIONS=False, compute_indices=True))

log.info("FULLRES_SMOKE_OK")
