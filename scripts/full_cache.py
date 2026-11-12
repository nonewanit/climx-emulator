"""Full-res cache BUILDER + UPLOADER — runs in a Kaggle CPU session.

CPU sessions do NOT consume GPU quota. This job builds everything the
training sessions need, ONCE:
  /tmp/full_cache/X_<var>_<scenario>.zarr   (fp16+zstd)
  /tmp/full_cache/y_<var>_<scenario>.zarr   (slab-read, RAM-safe)
  /tmp/full_cache/hist_<var>.zarr           (1961-1990 window — all the
                                             indices ever read)
  /tmp/full_cache/X_test.nc                 (7,300-day test forcings)
then uploads them in ~18 GB batches as PRIVATE Kaggle datasets
(climx-cache-b1, -b2, ...) and prints a manifest to commit.

GPU training sessions then attach those datasets and plain-copy them to
/tmp (sequential copies at the mount's ~26 MB/s) — no slow reads, no
re-preprocessing, no raw 171 GB input needed.

Run:  %run scripts/full_cache.py [--subsample 3]
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import xarray as xr

import climx_project  # noqa: F401
from climx_project import paths
from climx_project.cachebuild import (
    open_hist_from_cache, upload_batches, write_cache_x, write_cache_y,
    write_hist_window)
from climx_project.cacheio import zarr_fp16_encoding
from climx_project.data import daily_forcings, open_full_datasets_local
from src.data_preprocessing.preprocessing import preprocess_train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("full_cache")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

CACHE_DIR = Path(os.environ.get("CLIMX_CACHE_DIR", "/tmp/full_cache"))
MANIFEST = paths.PROJECT_ROOT / "data" / "full_cache_manifest.json"
OWNER = os.environ.get("CLIMX_KAGGLE_OWNER", "nonewanitchollakit")
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
CLIMATOLOGY = paths.PREPROCESSING_DIR / "climatology_full.nc"
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
    args = ap.parse_args()

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full_dir = Path(os.environ.get("CLIMX_FULL_DIR", str(find_full_dir())))
    log.info("full-res data dir: %s", full_dir)
    log.info("disk:\n%s", subprocess.run(["df", "-h", "/tmp", "/kaggle/working"],
                                         capture_output=True, text=True).stdout)

    ds = stage("open local full-res datasets", lambda: open_full_datasets_local(
        full_dir, scenarios=SCENARIOS))
    ds.hist = ds.hist.chunk(_HIST_CHUNKS)
    ds.train = ds.train.chunk(_CHUNKS)
    ds.hist = daily_forcings(ds.hist)

    # Frozen scaling params (none mode): bundled in the repo.
    if not SCALING_JSON.exists() and BUNDLED_SCALING.exists():
        SCALING_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED_SCALING, SCALING_JSON)
        log.info("using repo-bundled scaling params")
    if not SCALING_JSON.exists():
        raise RuntimeError("no scaling params — run the training script once or check assets/")

    ENCODING = zarr_fp16_encoding()

    # 1. Test forcings (once per session — CPU, ~10 min; ships in a batch).
    ds.test_forcings = ds.test_forcings.sel(time=slice("2081-01-01", "2101-01-01"))
    X_TEST_NC = CACHE_DIR / "X_test.nc"
    if not X_TEST_NC.exists():
        if not CLIMATOLOGY.exists():  # preprocess_test requires the file to exist
            xr.Dataset().to_netcdf(CLIMATOLOGY)
        from climx_project.preprocess import _save_chunked
        from src.data_preprocessing.preprocessing import preprocess_test
        X_test, _m = stage("preprocess_test (SSP2-4.5 forcings)", lambda: preprocess_test(
            ds, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
            scaling_params_path=SCALING_JSON, load_into_memory=False,
            stationarization_mode="none"))
        encoding = {v: {"zlib": False, "_FillValue": None} for v in X_test.data_vars}
        _save_chunked(X_test, X_TEST_NC, encoding=encoding)

    # 2. Training cache.
    ds.train = ds.train.isel(time=slice(0, None, args.subsample))
    X_scaled, _, _ = stage("pass B: lazy scaled fields", lambda: preprocess_train(
        ds, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
        scaling_params_path=SCALING_JSON, load_into_memory=False,
        stationarization_mode="none"))
    write_cache_x(X_scaled, CACHE_DIR, SCENARIOS, ENCODING)
    write_cache_y(full_dir, CACHE_DIR, SCENARIOS, ENCODING, SCALING_JSON)

    # 3. Hist window (the only hist the indices read).
    write_hist_window(ds.hist, CACHE_DIR, ENCODING)
    _ = open_hist_from_cache(CACHE_DIR)  # sanity: mergeable

    # 4. Upload in batches.
    manifest = upload_batches(CACHE_DIR, OWNER, MANIFEST)
    log.info("CACHE_UPLOAD_OK — %d dataset(s):", len(manifest))
    for slug, files in manifest.items():
        log.info("  %s: %s", slug, files)
    log.info("COMMIT data/full_cache_manifest.json to the repo:\n%s",
             json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
