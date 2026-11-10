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
import subprocess
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

# NOTE: paths.PREPROCESSING_DIR was resolved at `import climx_project` time
# (env overrides set after the import are ignored) — the small artifacts
# (scaling JSON, mode marker) land in the repo's data/preprocessing_data_lite/
# inside /kaggle/working, which persists across sessions of the notebook.
CACHE_DIR = Path(os.environ.get("CLIMX_CACHE_DIR", "/tmp/full_cache"))
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
CLIMATOLOGY = paths.PREPROCESSING_DIR / "climatology_full.nc"
MODE_MARKER = paths.PREPROCESSING_DIR / "stationarization_mode.txt"
# Scaling params are FROZEN constants (mean/std per channel) — committing them
# to the repo means no run ever recomputes them, even after /kaggle/working
# wipes or fresh notebook imports.
BUNDLED_SCALING = paths.PROJECT_ROOT / "assets" / "fullres_scaling_parameters.json"
SCENARIOS = ["ssp126", "ssp370", "ssp585"]


def _json_has_nan(path: Path) -> bool:
    """True if the scaling JSON contains NaN stats (a known failure mode of
    the monthly-mode scaling stage — observed mean=nan for every variable)."""
    import math
    if not path.exists():
        return False
    d = json.load(open(path))

    def walk(o) -> bool:
        if isinstance(o, float) and math.isnan(o):
            return True
        if isinstance(o, dict):
            return any(walk(v) for v in o.values())
        if isinstance(o, list):
            return any(walk(v) for v in o)
        return False
    return walk(d)

# Native-res forcing dims must be chunked BEFORE the daily upsample expands
# them (one full-var chunk would expand to ~60k days in a single dask task).
# time=-1 makes every daily variable ONE physical read: the mount does
# 26 MB/s sequentially but ~2 MB/s on scattered chunk reads (the y targets
# crawled at 20 min/var through 86 separate reads; one read = ~4-5 min/var).
# Peak memory: one 6.9 GB fp32 var at a time — fine on Kaggle GPU instances.
_CHUNKS = {"time": -1, "time_month": 24, "time_year": -1, "lat": -1, "lon": -1}
# The hist is read ONLY by the indices' base period (1961-1990, 4 vars): a
# -1 chunk would force whole-13.3 GB reads per var; 10-year chunks let the
# 30-year window read just its ~3 chunks per var (~13 GB total instead of 53).
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
    ap.add_argument("--stationarization", default="none", choices=("none", "monthly"),
                    help="none = standardized RAW fields (fast, no climatology; "
                         "the default). monthly = the organizer's anomaly pipeline "
                         "(climatology + de-stationarization).")
    ap.add_argument("--max-days", type=int, default=0,
                    help=">0 = smoke: truncate training days per scenario")
    args = ap.parse_args()

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full_dir = Path(os.environ.get("CLIMX_FULL_DIR", str(find_full_dir())))
    log.info("full-res data dir: %s", full_dir)
    df = subprocess.run(["df", "-h", "/tmp", "/kaggle/working"],
                        capture_output=True, text=True).stdout
    log.info("disk:\n%s", df)

    # 1. Open everything locally (metadata only, seconds).
    ds = stage("open local full-res datasets", lambda: open_full_datasets_local(
        full_dir, scenarios=SCENARIOS))
    log.info("hist: %s | train: %s | test: %s",
             dict(ds.hist.sizes), dict(ds.train.sizes), dict(ds.test_forcings.sizes))

    ds.hist = ds.hist.chunk(_HIST_CHUNKS)
    ds.train = ds.train.chunk(_CHUNKS)
    ds.hist = daily_forcings(ds.hist)

    # 2. Scaling params (once; persisted in /kaggle/working). In "none" mode
    # there is NO climatology: the params standardize the raw fields and the
    # whole anomaly machinery is skipped. In "monthly" mode the climatology
    # uses fast_monthly_climatology (resample-based, numerically identical to
    # the starter's ~12 h groupby; verified 1e-14).
    mode = args.stationarization
    if not SCALING_JSON.exists() and BUNDLED_SCALING.exists() and mode == "none":
        log.info("using repo-bundled scaling params (%s)", BUNDLED_SCALING)
        SCALING_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED_SCALING, SCALING_JSON)
        MODE_MARKER.write_text("none")
    pass_a_needed = (not SCALING_JSON.exists() or _json_has_nan(SCALING_JSON)
                     or not MODE_MARKER.exists()
                     or MODE_MARKER.read_text().strip() != mode)
    if pass_a_needed:
        from src.data_preprocessing.preprocessing import FORCING_VARIABLES, TARGET_VARIABLES  # noqa: F401
        if mode == "monthly":  # always recompute here: a none-mode placeholder may exist
            from climx_project.preprocess import fast_monthly_climatology
            hist_s = ds.hist.isel(time=slice(0, None, 3))
            with ProgressBar():
                tclim = stage("climatology: targets (fast resample)", lambda:
                              fast_monthly_climatology(hist_s[TARGET_VARIABLES]))
                fclim = stage("climatology: forcings (fast resample)", lambda:
                              fast_monthly_climatology(hist_s[FORCING_VARIABLES]))
            import dask
            clim = xr.merge([tclim, fclim])
            with dask.config.set(scheduler="threads"):
                clim = clim.compute()
            clim.to_netcdf(CLIMATOLOGY)
            log.info("wrote %s", CLIMATOLOGY)
        tiny = type("DS", (), {
            "hist": ds.hist.isel(time=slice(0, None, 3)),
            "train": ds.train.isel(time=slice(0, 2000)),
            "test_forcings": ds.test_forcings,
        })()
        with ProgressBar():
            stage(f"scaling params (2000-day sample, {mode})",
                  lambda: preprocess_train(
                      tiny, preprocessing_path=paths.PREPROCESSING_DIR,
                      version="full", scaling_params_path=None,
                      load_into_memory=False, stationarization_mode=mode))
        if _json_has_nan(SCALING_JSON):
            raise RuntimeError("scaling params contain NaN — aborting")
        MODE_MARKER.write_text(mode)
        BUNDLED_SCALING.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCALING_JSON, BUNDLED_SCALING)
        log.info("scaling params copied to %s — COMMIT this file so no run ever recomputes them",
                 BUNDLED_SCALING)
    else:
        log.info("scaling params exist for mode %s — skipping pass A", mode)

    # 3. Test preprocessing. X_test.nc lives in /tmp (3.4 GB — /kaggle/working
    # is a 20 GB FUSE mount that fills up and kills big writes; /tmp has
    # ~1 TB). It re-runs each session (~10 min) — acceptable.
    # The test forcings file spans 2015-2100; the submission window is
    # 2081-2100 (7,300 NoLeap days). Slice BEFORE preprocessing — the full
    # file would produce a ~152 GB X_test.nc.
    ds.test_forcings = ds.test_forcings.sel(time=slice("2081-01-01", "2101-01-01"))
    X_TEST_NC = Path(os.environ.get("CLIMX_XTEST_DIR", "/tmp")) / "X_test.nc"

    def _metadata_for():
        return {
            "climatology_path": str(CLIMATOLOGY),
            "scaling_params_path": str(SCALING_JSON),
            "scaling_params": json.load(open(SCALING_JSON)),
            "stationarization_mode": mode,
        }

    if X_TEST_NC.exists():
        try:
            X_test = xr.open_dataset(X_TEST_NC, chunks={"time": 512, "lat": -1, "lon": -1})
            _ = dict(X_test.sizes)  # force open; corrupt files raise here
            metadata_test = _metadata_for()
        except Exception:
            log.warning("X_test.nc is corrupt (partial write) — removing and redoing")
            X_TEST_NC.unlink()
    if not X_TEST_NC.exists():
        # preprocess_test hard-requires the climatology file to EXIST even in
        # none-mode (it is never read there) — drop an empty placeholder.
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
        encoding = {var: {"zlib": False, "_FillValue": None} for var in X_test.data_vars}
        _save_chunked(X_test, X_TEST_NC, encoding=encoding)
    log.info("X_test: %s (stationarization=%s)", dict(X_test.sizes), mode)

    # 4. Write the subsampled fp16 training cache to /tmp (per session).
    ds.train = ds.train.isel(time=slice(0, None, args.subsample))
    X_scaled, y_scaled, _ = stage(
        "pass B: lazy scaled anomalies (subsampled train)",
        lambda: preprocess_train(
            ds, preprocessing_path=paths.PREPROCESSING_DIR, version="full",
            scaling_params_path=SCALING_JSON, load_into_memory=False,
            stationarization_mode=mode))

    from climx_project.cacheio import zarr_fp16_encoding
    ENCODING = zarr_fp16_encoding()

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
            # ONE compute per variable: the chunked lazy path made hundreds of
            # small scattered reads per variable, which the network-backed
            # input mount serves at ~1 MB/s (dd shows 26 MB/s sequential, so
            # it's a pattern penalty, not bandwidth). Compute the whole var
            # into memory once (~1.2 GB fp16), then write to local zarr with
            # bounded chunks for the training-side block reads.
            da = dsx.sel(forcing_scenario=scen)[name]
            # 1D global vars (yearly GHGs) must become full fields: the U-Net
            # treats every channel as (time, lat, lon) — the starter's own
            # pipeline broadcasts them; broadcast here so the cache is uniform.
            if "lat" not in da.dims:
                da = da.expand_dims(lat=dsx.lat, lon=dsx.lon).transpose("time", "lat", "lon")
            # ONE sequential read pass: rechunking to a single time chunk makes
            # dask read the whole variable contiguously. The mount does 26 MB/s
            # sequentially (dd) but ~2 MB/s on scattered chunk reads — the y
            # (daily target) vars were crawling at 22 min each through 29
            # separate 80 MB chunk reads.
            da = da.chunk({"time": -1})
            arr = da.astype("float16").compute().values  # DataArray.compute() -> DataArray; need numpy
            ds_out = xr.Dataset({name: (da.dims, arr)}, coords=da.coords)
            # 1D vars (yearly GHGs) have no lat/lon — chunk only the dims
            # that exist; 128-day chunks keep the training block reads cheap.
            chunks = {d: (128 if d == "time" else -1) for d in da.dims}
            ds_out.chunk(chunks).to_zarr(
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
    # Model + checkpoints + predictions + indices all live in /tmp (the
    # predictions zarr alone is ~11 GB — impossible on the 20 GB mount).
    stage("fit (cache, 192x288)", lambda: emulator.fit(
        CACHE_DIR, None, trainer_params=dict(
            max_epochs=args.epochs, batch_size=args.batch, num_workers=2,
            sample_step=1, scenarios=SCENARIOS, max_days=args.max_days,
            accelerator="auto", devices=1,
            model_path="/tmp/full_unet_models",
            logger=False)))
    emulator.save("/tmp/full_unet.pt")

    # 6. Evaluate: predict + reversal + FULL index computation + submission.
    # Threads scheduler for this stage: everything it touches is local/in-
    # memory (X_test.nc, in-memory predictions, lazy hist) — no shared remote
    # handles — and the 15 xclim indices parallelize ~4x. The zarr prediction
    # write goes through zarr v3's own async machinery, not the WSL-era
    # dask.array.store path that deadlocked on this machine.
    import dask
    with dask.config.set(scheduler="threads"):
        y_phys, indices_pred = stage(
            "evaluate (indices ON, SSP2-4.5 2081-2100)", lambda: emulator.evaluate(
                X_test, metadata_test,
                results_path=Path("/tmp/full_unet_evaluation.json"),
                predictions_path=Path("/tmp/full_unet_predictions.zarr"),
                historical_data_path=ds.hist, TARGET_VARIABLES=paths.TARGET_VARIABLES,
                LOAD_PREDICTIONS=False, compute_indices=True))
    parquet = stage("build submission parquet", lambda: submission.build_submission(
        indices_pred, "unet_full"))
    report = submission.verify_parquet(parquet)
    log.info("parquet verified: %s", report)
    log.info("FULLRES_TRAIN_OK — %s", parquet)


if __name__ == "__main__":
    main()
