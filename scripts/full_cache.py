"""One-time full-res preprocessing cache — runs on Kaggle (interactive notebook).

Streams the REAL full-res data once and writes compacted, reusable caches:
  data/preprocessing_data_full/climatology_full.nc        (monthly climatology)
  data/preprocessing_data_full/scaling_parameters_full.json
  /tmp/full_cache/X_<var>_<scenario>.zarr, y_<var>_<scenario>.zarr
                                                          (fp16 + zstd chunks)

Scenario-pure files: Phase C (world model) needs contiguous daily sequences
WITHIN a scenario, and training can attach only the scenario(s) it wants.

Disk reality on Kaggle: /kaggle/working is 20 GB but the root overlay has
~1 TB free, so zarr files are written to /tmp (CLIMX_CACHE_DIR) and then
AUTO-UPLOADED in ~20 GB batches as PRIVATE Kaggle datasets (the CLI only
uploads whole folders, and each dataset gets its own size allowance).
Completed batches are deleted from /tmp immediately.

RESUMABLE across sessions via a manifest persisted in /kaggle/working
(data/full_cache_meta/manifest.json): a 9 h session kill loses at most the
batch in flight (and the manifest is saved after every upload). Re-run the
script until it prints CACHE_OK — then commit the printed manifest into the
repo (docs/full_cache_manifest.json) so the training notebook knows which
datasets to attach.

Run:  %run scripts/full_cache.py [--subsample 1|3]
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
from climx_project import paths
from climx_project.data import daily_forcings, open_full_datasets
from src.data_preprocessing.preprocessing import preprocess_train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("full_cache")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

os.environ.setdefault("CLIMX_PREPROCESS_DIR", "/kaggle/working/data/preprocessing_data_full")
CACHE_DIR = Path(os.environ.get("CLIMX_CACHE_DIR", "/tmp/full_cache"))
META_DIR = Path(os.environ.get("CLIMX_CACHE_META_DIR", "/kaggle/working/data/full_cache_meta"))
MANIFEST_PATH = META_DIR / "manifest.json"
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
CLIMATOLOGY = paths.PREPROCESSING_DIR / "climatology_full.nc"

BATCH_GB = float(os.environ.get("CLIMX_BATCH_GB", "20"))
OWNER = os.environ.get("CLIMX_KAGGLE_OWNER", "nonewanitchollakit")

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


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"uploaded": [], "batch_idx": 0, "batches": {}}


def save_manifest(m: dict) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(m, f, indent=2)


def upload_batch(files: list[Path], prefix: str, manifest: dict) -> None:
    """Upload `files` (+ the small preprocessing outputs on batch 1) as a new
    private dataset, then delete them from /tmp."""
    idx = int(manifest.get("batch_idx", 0)) + 1
    slug = f"{OWNER}/climx-cache-{prefix.lower()}-{idx}"
    stage_dir = Path("/tmp/ds_stage")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for p in files:
        shutil.copytree(p, stage_dir / p.name)
    # The climatology + scaling JSON must reach the TRAINING notebook (whose
    # /kaggle/working is separate) — ship them in the first batch.
    if idx == 1:
        for p in (CLIMATOLOGY, SCALING_JSON):
            if p.exists():
                shutil.copy2(p, stage_dir / p.name)
    metadata = {"title": f"climx-full-cache-{prefix.lower()}-{idx}",
                "id": slug,
                "licenses": [{"name": "CC-BY-4.0"}]}
    with open(stage_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    gb = sum(p.stat().st_size for p in stage_dir.rglob("*") if p.is_file()) / 1e9
    log.info("=== upload batch %d (%s): %.1f GB -> %s ===", idx, prefix, gb, slug)
    t0 = time.time()
    subprocess.run(["kaggle", "datasets", "create", "-p", str(stage_dir), "--private"],
                   check=True)
    mins = (time.time() - t0) / 60
    log.info("OK upload %s: %.1f min, %.1f GB/min", slug, mins, gb / mins)

    manifest["uploaded"] += [p.name for p in files]
    manifest["batch_idx"] = idx
    manifest.setdefault("batches", {})[slug] = [p.name for p in files]
    save_manifest(manifest)
    for p in files:
        shutil.rmtree(p)
    shutil.rmtree(stage_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=1,
                    help="keep every Nth training day in the cache "
                         "(1 = Phase C-ready contiguous daily; 3 = 1/3 size). "
                         "Reads are chunk-granular, so this barely changes the "
                         "job duration — it changes cache SIZE.")
    args = ap.parse_args()

    paths.ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    uploaded = set(manifest.get("uploaded", []))
    log.info("manifest: %d uploaded so far", len(uploaded))

    # 1. Open everything lazily (metadata only, ~30 s).
    ds = stage("open full-res datasets (real hist + train)", lambda: open_full_datasets())
    log.info("hist: %s | train: %s", dict(ds.hist.sizes), dict(ds.train.sizes))

    # Chunk the native-res forcing dims BEFORE any daily expansion.
    ds.hist = ds.hist.chunk(_CHUNKS)
    ds.train = ds.train.chunk(_CHUNKS)
    if args.subsample != 1:
        ds.train = ds.train.isel(time=slice(0, None, args.subsample))

    # The climatology path needs DAILY time on the hist forcings (KeyError
    # 'time.month' otherwise); the train forcings are upsampled internally by
    # the starter's preprocess_train.
    ds.hist = daily_forcings(ds.hist)

    # 2. Pass A: climatology + scaling params (small files, one-time).
    # Fast resample-based climatology (numerically identical to the starter's
    # ~12 h groupby; verified 1e-14). Threads safe on local data.
    if CLIMATOLOGY.exists() and SCALING_JSON.exists():
        log.info("climatology + scaling params exist — skipping pass A")
    else:
        if not CLIMATOLOGY.exists():
            from climx_project.preprocess import fast_monthly_climatology
            from src.data_preprocessing.preprocessing import FORCING_VARIABLES, TARGET_VARIABLES
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
            stage("scaling params (2000-day sample)",
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

    # 4. Write each (variable, scenario) to its own fp16+zstd zarr file,
    # uploading in ~BATCH_GB batches per scenario (resumable via manifest).
    # Scenario-pure files: Phase C needs contiguous daily sequences WITHIN a
    # scenario, and training can attach only the scenario(s) it wants.
    SCENARIOS = ["ssp126", "ssp370"]
    from climx_project.cacheio import zarr_fp16_encoding
    ENCODING = zarr_fp16_encoding()

    def write_vars(dsx, prefix):
        plan = []
        for scen in SCENARIOS:
            for name in dsx.data_vars:
                key = f"{prefix}_{name}_{scen}"
                if key in uploaded:
                    continue
                gb = float(dsx[name].size) * 2 / 1e9 / len(SCENARIOS)
                plan.append((scen, name, key, gb))
        total_gb = sum(g for *_, g in plan)
        log.info("vars to write (%s): %d — %.1f GB total", prefix, len(plan), total_gb)
        pending: list[Path] = []
        pending_gb = 0.0
        done_gb = 0.0
        rate = None
        scen_ds = None
        current_scen = None
        for i, (scen, name, key, gb) in enumerate(plan):
            if scen != current_scen:
                if pending:  # flush on scenario switch: datasets stay scenario-pure
                    upload_batch(pending, f"{current_scen}-{prefix.lower()}", manifest)
                    pending, pending_gb = [], 0.0
                current_scen = scen
                scen_ds = dsx.sel(forcing_scenario=scen)
            final = CACHE_DIR / f"{key}.zarr"
            tmp = CACHE_DIR / f".tmp_{key}.zarr"
            if tmp.exists():
                shutil.rmtree(tmp)
            log.info("=== write %s (%d/%d, %.1f GB) ===", key, i + 1, len(plan), gb)
            t0 = time.time()
            with ProgressBar():
                scen_ds[[name]].astype("float16").to_zarr(tmp, mode="w",
                                                          encoding={name: ENCODING})
            tmp.rename(final)
            mins = (time.time() - t0) / 60
            rate = gb / mins
            done_gb += gb
            pending.append(final)
            pending_gb += gb
            eta_h = (total_gb - done_gb) / rate / 60 if rate else float("nan")
            log.info("OK %s: %.1f min, %.2f GB/min | done %.1f/%.1f GB | ETA %.1f h",
                     key, mins, rate, done_gb, total_gb, eta_h)
            if pending_gb >= BATCH_GB:
                upload_batch(pending, f"{scen}-{prefix.lower()}", manifest)
                pending, pending_gb = [], 0.0
        if pending:
            upload_batch(pending, f"{current_scen}-{prefix.lower()}", manifest)

    write_vars(X_scaled, "X")
    write_vars(y_scaled, "y")
    batches = list(manifest.get("batches", {}).keys())
    log.info("CACHE_OK — %d dataset(s): %s", len(batches), batches)
    log.info("commit this manifest into docs/full_cache_manifest.json:\n%s",
             json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
