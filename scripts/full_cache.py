"""One-time full-res preprocessing cache — runs on Kaggle (interactive notebook).

Streams the REAL full-res data once and writes compacted, reusable caches:
  data/preprocessing_data_full/climatology_full.nc        (monthly climatology)
  data/preprocessing_data_full/scaling_parameters_full.json
  data/full_cache/X_<var>.zarr, y_<var>.zarr             (fp16 + zstd chunks)

The cache is what Phase B (U-Net training) and Phase C (world model) both
read. Two-pass design to minimize streaming:

  pass A (once): climatology from the historical files + scaling params from
                 a 2000-day train sample — small files, ~one session's
                 streaming of the hist targets.
  pass B:        the starter's preprocess_train with the cached scaling JSON
                 (skips the stats pass) returns lazy scaled anomalies; each
                 variable is then written to its own zarr file.

RESUMABLE: every variable has its own file, written to a .tmp name and
renamed on completion. A 9 h session kill loses at most the variable in
flight — just re-run the script until it prints CACHE_OK. Throughput is
logged per variable so we learn the real HF streaming rate.

Run:  %run scripts/full_cache.py [--subsample 1|3]
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

import climx_project  # noqa: F401  (vendor + env + compat patches)
from climx_project import paths
from climx_project.data import daily_forcings, open_full_datasets
from src.data_preprocessing.preprocessing import preprocess_train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("full_cache")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

os.environ.setdefault("CLIMX_PREPROCESS_DIR", "/kaggle/working/data/preprocessing_data_full")
CACHE_DIR = Path(os.environ.get("CLIMX_CACHE_DIR", "/kaggle/working/data/full_cache"))
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
CLIMATOLOGY = paths.PREPROCESSING_DIR / "climatology_full.nc"

# Native-res forcing dims must be chunked BEFORE the daily upsample expands
# them: one full-var chunk would expand to 60k days x 192x288 in a single
# dask task (~14-66 GB RAM — OOM on Kaggle). ~24 months per chunk keeps the
# per-task expansion at a few hundred MB.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=1,
                    help="keep every Nth training day in the cache "
                         "(1 = Phase C-ready contiguous daily; 3 = 1/3 size). "
                         "Reads are chunk-granular, so this barely changes the "
                         "job duration — it changes cache SIZE.")
    args = ap.parse_args()
    step = args.subsample

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Open everything lazily (metadata only, ~30 s).
    ds = stage("open full-res datasets (real hist + train)", lambda: open_full_datasets())
    log.info("hist: %s | train: %s", dict(ds.hist.sizes), dict(ds.train.sizes))

    # Chunk the native-res forcing dims BEFORE any daily expansion.
    ds.hist = ds.hist.chunk(_CHUNKS)
    ds.train = ds.train.chunk(_CHUNKS)
    if args.subsample != 1:
        ds.train = ds.train.isel(time=slice(0, None, step))

    # The climatology path needs DAILY time on the hist forcings (KeyError
    # 'time.month' otherwise); the train forcings are upsampled internally by
    # the starter's preprocess_train.
    ds.hist = daily_forcings(ds.hist)

    # 2. Pass A: climatology + scaling params (small files, one-time).
    if CLIMATOLOGY.exists() and SCALING_JSON.exists():
        log.info("climatology + scaling params exist — skipping pass A")
    else:
        hist_s = ds.hist.isel(time=slice(0, None, 3))  # cheaper groupby compute
        tiny = type("DS", (), {
            "hist": hist_s,
            "train": ds.train.isel(time=slice(0, 2000)),
            "test_forcings": ds.test_forcings,
        })()
        with ProgressBar():
            stage("pass A: climatology + scaling params (hist + 2000-day sample)",
                  lambda: preprocess_train(
                      tiny, preprocessing_path=paths.PREPROCESSING_DIR,
                      version="full", scaling_params_path=None,
                      load_into_memory=False, stationarization_mode="monthly"))

    # 3. Pass B: lazy scaled anomalies over the full train (no stats pass).
    X_scaled, y_scaled, _meta = stage(
        "pass B: lazy scaled anomalies (full train)",
        lambda: preprocess_train(
            ds, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
            scaling_params_path=SCALING_JSON, load_into_memory=False,
            stationarization_mode="monthly"))
    log.info("X vars: %s", sorted(list(X_scaled.data_vars)))

    # 4. Write each variable to its own fp16+zstd zarr file (resumable).
    from numcodecs import Blosc
    ENCODING = {"dtype": "float16",
                "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)}

    def write_vars(dsx, prefix):
        names = [n for n in dsx.data_vars
                 if not (CACHE_DIR / f"{prefix}_{n}.zarr").exists()]
        total_gb = sum(float(dsx[n].size) * 2 / 1e9 for n in names)
        log.info("vars to write (%s): %d — %.1f GB total",
                 prefix, len(names), total_gb)
        done_gb = 0.0
        rate = None
        for i, name in enumerate(names):
            final = CACHE_DIR / f"{prefix}_{name}.zarr"
            tmp = CACHE_DIR / f".tmp_{prefix}_{name}.zarr"
            if tmp.exists():
                shutil.rmtree(tmp)
            gb = float(dsx[name].size) * 2 / 1e9
            log.info("=== write %s (%d/%d, %.1f GB) ===", name, i + 1, len(names), gb)
            t0 = time.time()
            with ProgressBar():
                dsx[[name]].astype("float16").to_zarr(tmp, mode="w",
                                                      encoding={name: ENCODING})
            tmp.rename(final)
            mins = (time.time() - t0) / 60
            rate = gb / mins
            done_gb += gb
            eta_h = (total_gb - done_gb) / rate / 60 if rate else float("nan")
            log.info("OK %s: %.1f min, %.2f GB/min | done %.1f/%.1f GB | ETA %.1f h",
                     name, mins, rate, done_gb, total_gb, eta_h)

    write_vars(X_scaled, "X")
    write_vars(y_scaled, "y")
    log.info("CACHE_OK")


if __name__ == "__main__":
    main()
