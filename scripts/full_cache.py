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

Run:  %run scripts/full_cache.py [--subsample 1]  (default 1 = Phase B+C)
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
    ap.add_argument("--subsample", type=int, default=1)
    args = ap.parse_args()

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full_dir = Path(os.environ.get("CLIMX_FULL_DIR", str(find_full_dir())))
    log.info("full-res data dir: %s", full_dir)
    log.info("disk:\n%s", subprocess.run(["df", "-h", "/tmp", "/kaggle/working"],
                                         capture_output=True, text=True).stdout)

    # The mount is used ONLY for lazy metadata opens (tiny reads — fast).
    # Raw data arrives via hf downloads PER SCENARIO and is deleted after
    # use: the CPU session's overlay is small (~100-150 GB), so the raw
    # files and the cache must never coexist at full size.
    ds = stage("open mount datasets (metadata only)", lambda: open_full_datasets_local(
        full_dir, scenarios=SCENARIOS))
    ds.hist = ds.hist.chunk(_HIST_CHUNKS)

    # Frozen scaling params (none mode): bundled in the repo.
    if not SCALING_JSON.exists() and BUNDLED_SCALING.exists():
        SCALING_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED_SCALING, SCALING_JSON)
        log.info("using repo-bundled scaling params")
    if not SCALING_JSON.exists():
        raise RuntimeError("no scaling params — run the training script once or check assets/")

    ENCODING = zarr_fp16_encoding()

    def uploaded_set() -> set:
        m = {}
        if MANIFEST.exists():
            m = json.load(open(MANIFEST))
        return {f for files in m.values() for f in files}

    def maybe_upload():
        # upload completed files once ~18 GB has accumulated — a session kill
        # loses at most the un-uploaded remainder (~one batch)
        def _sz(p):
            if p.is_dir():
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            return p.stat().st_size
        pending = sum(_sz(p) for p in CACHE_DIR.iterdir()
                      if (p.name.endswith(".zarr") or p.name == "X_test.nc")
                      and not p.name.startswith(".tmp"))
        if pending >= 18e9:
            upload_batches(CACHE_DIR, OWNER, MANIFEST)

    def download_one(name: str) -> Path:
        """hf download to /tmp (xet backend, ~60 MB/s measured) — one file
        at a time so the CPU session's small overlay never holds everything."""
        from huggingface_hub import hf_hub_download
        dst = Path("/tmp/raw_full") / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 1e9:
            return dst
        t0 = time.time()
        p = hf_hub_download(repo_id="isp-uv-es/ClimX", repo_type="dataset",
                            filename=f"full/{name}",
                            local_dir=str(Path("/tmp/hf_tmp")))
        shutil.move(p, dst)
        log.info("OK hf download %s (%.1f min)", name, (time.time() - t0) / 60)
        return dst

    PREFIX = "NorESM2-MM_r1i1p1f1_with_forcing_"

    # 1. Test forcings (download the small file; local preprocess_test).
    test_raw = download_one(f"{PREFIX}test_forcings_ssp245.nc")
    ds.test_forcings = xr.open_dataset(test_raw, engine="h5netcdf",
                                       chunks=_CHUNKS).sel(
        time=slice("2081-01-01", "2101-01-01"))
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
        # The aerosol split vars inherit float64 from the source (X_test.nc
        # came out 30.7 GB) — fp16 matches the training cache and shrinks it ~4x.
        X_test = X_test.astype("float16")
        encoding = {v: {"zlib": False, "_FillValue": None} for v in X_test.data_vars}
        _save_chunked(X_test, X_TEST_NC, encoding=encoding)
    test_raw.unlink()

    # 2. Per-scenario streaming: download -> build X+y -> upload -> delete.
    for scen in SCENARIOS:
        raw = download_one(f"{PREFIX}train_{scen}.nc")
        one = type("DS", (), {
            "hist": ds.hist,
            "train": xr.open_dataset(raw, engine="h5netcdf", chunks=_CHUNKS)
                      .isel(time=slice(0, None, args.subsample)),
            "test_forcings": ds.test_forcings,
        })()
        X_scaled, _, _ = stage(f"pass B ({scen})", lambda one=one: preprocess_train(
            one, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
            scaling_params_path=SCALING_JSON, load_into_memory=False,
            stationarization_mode="none"))
        write_cache_x(X_scaled, CACHE_DIR, [scen], ENCODING,
                      skip=uploaded_set(), flush=maybe_upload)
        write_cache_y(raw.parent, CACHE_DIR, [scen], ENCODING, SCALING_JSON,
                      skip=uploaded_set(), flush=maybe_upload)
        upload_batches(CACHE_DIR, OWNER, MANIFEST)
        raw.unlink()
        log.info("deleted %s", raw.name)

    # 3. Hist window — 1961-1990 lies entirely inside the second hist file.
    hist_raw = download_one(f"{PREFIX}hist_train_1932_2014.nc")
    hist_ds = xr.open_dataset(hist_raw, engine="h5netcdf", chunks=_HIST_CHUNKS)
    write_hist_window(hist_ds, CACHE_DIR, ENCODING, skip=uploaded_set(),
                      flush=maybe_upload)
    upload_batches(CACHE_DIR, OWNER, MANIFEST)
    hist_raw.unlink()

    # 4. Remaining files (e.g. X_test.nc) if any.
    manifest = upload_batches(CACHE_DIR, OWNER, MANIFEST)
    log.info("CACHE_UPLOAD_OK — %d dataset(s):", len(manifest))
    for slug, files in manifest.items():
        log.info("  %s: %s", slug, files)
    log.info("COMMIT data/full_cache_manifest.json to the repo:\n%s",
             json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
