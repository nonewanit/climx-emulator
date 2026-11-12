"""Cache build / transfer / upload shared by the full-res scripts.

Architecture (Nov 12): a CPU session builds the training cache once
(X + y fp16 zarr + X_test.nc + a 1961-1990 hist window) and uploads it
in ~18 GB batches as private Kaggle datasets; GPU training sessions
attach those datasets and plain-copy them to /tmp (sequential copies
run at the mount's ~26 MB/s). No slow reads in training sessions at all.
"""

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

TARGETS = ["huss", "pr", "psl", "sfcWind", "tas", "tasmax", "tasmin"]
FULL_PREFIX = "NorESM2-MM_r1i1p1f1_with_forcing_"


def write_cache_x(X_scaled, cache_dir: Path, scenarios, encoding: dict,
                  skip: set | None = None, flush=None) -> None:
    """X cache: one compute per (scenario, var); 1D GHGs broadcast to the
    grid; 128-day output chunks. X sources are small (monthly/yearly) so
    this is fast even through the mount. `skip` = keys already uploaded
    (session-resume via the manifest). `flush` = optional callback called
    after each completed variable (uploads pending files in ~18 GB batches
    so a session kill loses at most one batch)."""
    skip = skip or set()
    plan = [(s, n) for s in scenarios for n in X_scaled.data_vars]
    for i, (scen, name) in enumerate(plan):
        key = f"X_{name}_{scen}"
        if key in skip:
            continue
        out = cache_dir / f"{key}.zarr"
        if out.exists():
            continue
        tmp = cache_dir / f".tmp_{key}.zarr"
        if tmp.exists():
            shutil.rmtree(tmp)
        log.info("=== write %s (%d/%d) ===", key, i + 1, len(plan))
        t0 = time.time()
        da = X_scaled.sel(forcing_scenario=scen)[name]
        if "lat" not in da.dims:
            da = da.expand_dims(lat=X_scaled.lat, lon=X_scaled.lon).transpose("time", "lat", "lon")
        arr = da.compute().values.astype("float16")
        ds_out = xr.Dataset({name: (da.dims, arr)}, coords=da.coords)
        chunks = {d: (128 if d == "time" else -1) for d in da.dims}
        ds_out.chunk(chunks).to_zarr(tmp, mode="w", encoding={name: encoding})
        tmp.rename(out)
        log.info("OK %s (%.1f min)", key, (time.time() - t0) / 60)
        if flush:
            flush()


def write_cache_y(full_dir: Path, cache_dir: Path, scenarios,
                  encoding: dict, scaling_json: Path,
                  skip: set | None = None, flush=None) -> None:
    """y cache: SLAB-BASED direct netCDF4 reads (1,000-day contiguous
    hyperslabs — the mount's fast pattern — with ~220 MB transient memory).
    Transforms match the starter exactly: log10(x+eps) for pr/huss, then
    z-score with the frozen params."""
    from netCDF4 import Dataset as NCDataset

    sc = json.load(open(scaling_json))["scaling_params"]
    SLAB = 1000
    skip = skip or set()
    plan = [(s, v) for s in scenarios for v in TARGETS]
    for i, (scen, var) in enumerate(plan):
        key = f"y_{var}_{scen}"
        if key in skip:
            continue
        out = cache_dir / f"{key}.zarr"
        if out.exists():
            continue
        tmp = cache_dir / f".tmp_{key}.zarr"
        if tmp.exists():
            shutil.rmtree(tmp)
        log.info("=== slab read %s (%d/%d) ===", key, i + 1, len(plan))
        t0 = time.time()
        slabs = []
        with NCDataset(str(full_dir / f"{FULL_PREFIX}train_{scen}.nc")) as nc:
            n = len(nc["time"])
            time_c = nc["time"][::3]
            lat = nc["lat"][:]
            lon = nc["lon"][:]
            for s0 in range(0, n, SLAB):
                s = nc[var][s0:s0 + SLAB][::3].astype(np.float32)
                if var in ("pr", "huss"):
                    s = np.log10(s + 1e-6)
                s = (s - sc[var]["mean"]) / sc[var]["std"]
                slabs.append(s.astype("float16"))
        arr = np.concatenate(slabs, axis=0)
        del slabs
        da = xr.DataArray(arr, dims=("time", "lat", "lon"),
                          coords={"time": time_c, "lat": lat, "lon": lon},
                          name=var)
        ds_out = xr.Dataset({var: da})
        ds_out.chunk({"time": 128, "lat": -1, "lon": -1}).to_zarr(
            tmp, mode="w", encoding={var: encoding})
        tmp.rename(out)
        log.info("OK %s (%.1f min)", key, (time.time() - t0) / 60)
        if flush:
            flush()


def write_hist_window(hist_ds, cache_dir: Path, encoding: dict,
                      lo: str = "1961-01-01", hi: str = "1990-12-31",
                      margin_days: int = 10, skip: set | None = None,
                      flush=None) -> None:
    """fp16 hist targets over the indices' base period (+margin for the
    doy window) — the ONLY hist the evaluate ever reads. ~5-6 GB total."""
    sel = hist_ds.sel(time=slice(lo, hi))
    t = hist_ds.time.values
    lo_v = sel.time.values[0]
    hi_v = sel.time.values[-1]
    i0 = max(0, int(np.flatnonzero(t >= lo_v)[0]) - margin_days)
    i1 = min(len(t), int(np.flatnonzero(t <= hi_v)[-1]) + margin_days + 1)
    win = hist_ds.isel(time=slice(i0, i1))
    skip = skip or set()
    for var in TARGETS:
        if f"hist_{var}" in skip:
            continue
        out = cache_dir / f"hist_{var}.zarr"
        if out.exists():
            continue
        tmp = cache_dir / f".tmp_hist_{var}.zarr"
        if tmp.exists():
            shutil.rmtree(tmp)
        log.info("=== write hist window %s ===", var)
        da = win[var]
        arr = da.compute().values.astype("float16")
        ds_out = xr.Dataset({var: (da.dims, arr)}, coords=da.coords)
        ds_out.chunk({"time": 128, "lat": -1, "lon": -1}).to_zarr(
            tmp, mode="w", encoding={var: encoding})
        tmp.rename(out)
        if flush:
            flush()
    log.info("hist window written")


def open_hist_from_cache(cache_dir: Path) -> xr.Dataset:
    """Merge the cached hist_*.zarr window into a Dataset for evaluate()."""
    ds = xr.Dataset()
    for var in TARGETS:
        p = cache_dir / f"hist_{var}.zarr"
        if p.exists():
            ds[var] = xr.open_zarr(p)[var]
    if not ds.data_vars:
        raise FileNotFoundError(f"no hist_*.zarr under {cache_dir}")
    return ds


def upload_batches(cache_dir: Path, owner: str, manifest_path: Path,
                   batch_gb: float = 18.0) -> dict:
    """Upload cache files in ~batch_gb batches as private Kaggle datasets
    (climx-cache-b1, -b2, ...), DELETING each batch from /tmp afterwards.

    This interleaves with the build (call it after each phase): the
    subsample-1 cache is ~170-200 GB — too big for one dataset and for one
    9 h session — and a session kill only loses the batch in flight.
    The manifest ({slug: [stems]}) persists at manifest_path; earlier
    sessions' batches are skipped."""
    manifest = {}
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
    uploaded = {f for files in manifest.values() for f in files}
    files = sorted([p for p in cache_dir.iterdir()
                    if (p.name.endswith(".zarr") or p.name == "X_test.nc")
                    and (p.stem if p.suffix == ".zarr" else p.name) not in uploaded
                    and not p.name.startswith(".tmp")],
                   key=lambda p: p.name)
    batch, batch_size = [], 0.0
    for p in files:
        gb = p.stat().st_size / 1e9
        if batch and batch_size + gb > batch_gb:
            _upload_one(batch, owner, manifest_path, manifest)
            batch, batch_size = [], 0.0
        batch.append(p)
        batch_size += gb
    if batch:
        _upload_one(batch, owner, manifest_path, manifest)
    return manifest


def _upload_one(batch: list[Path], owner: str, manifest_path: Path, manifest: dict) -> None:
    idx = len(manifest) + 1
    slug = f"{owner}/climx-cache-b{idx}"
    stage = Path("/tmp/upload_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for p in batch:
        shutil.copy2(p, stage / p.name)
    meta = {"title": f"climx-cache-b{idx}", "id": slug,
            "licenses": [{"name": "CC-BY-4.0"}]}
    with open(stage / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    gb = sum(p.stat().st_size for p in batch) / 1e9
    log.info("=== upload batch %d (%s): %.1f GB -> %s ===",
             idx, [p.name for p in batch], gb, slug)
    t0 = time.time()
    subprocess.run(["kaggle", "datasets", "create", "-p", str(stage), "--private"],
                   check=True)
    log.info("OK upload %s (%.1f min)", slug, (time.time() - t0) / 60)
    manifest[slug] = [p.stem if p.suffix == ".zarr" else p.name for p in batch]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    for p in batch:
        p.unlink()
    shutil.rmtree(stage)


def copy_raw_to_tmp(full_dir: Path, dest: Path) -> Path:
    """Fetch the needed raw NetCDFs to /tmp — huggingface_hub FIRST.

    Measured on Kaggle: hf_hub_download pulls a 33 GB file in 10-20 min
    (~30-55 MB/s) — faster than the mount's sequential cp (26 MB/s) and
    infinitely faster than library reads (~1-2 MB/s). After the download
    every read is local disk at ~1 GB/s. Falls back to a mount cp if HF
    fails. Resumable within a session: existing files skipped.
    """
    dest.mkdir(parents=True, exist_ok=True)
    needed = [f"{FULL_PREFIX}{s}.nc" for s in (
        "hist_train_1850_1931", "hist_train_1932_2014",
        "train_ssp126", "train_ssp370", "train_ssp585",
        "test_forcings_ssp245")]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    hf_tmp = Path("/tmp/hf_tmp")
    hf_tmp.mkdir(parents=True, exist_ok=True)
    for name in needed:
        dst = dest / name
        if dst.exists() and dst.stat().st_size > 1e9:  # >1 GB = complete-ish
            continue
        try:
            t0 = time.time()
            p = hf_hub_download(
                repo_id="isp-uv-es/ClimX", repo_type="dataset",
                filename=f"full/{name}", local_dir=str(hf_tmp))
            shutil.move(p, dst)
            log.info("OK hf download %s (%.1f min)", name, (time.time() - t0) / 60)
        except Exception as e:
            log.warning("hf download failed (%s) — falling back to mount cp", e)
            src = full_dir / name
            if not src.exists():
                raise FileNotFoundError(
                    f"{name} not on HF (download failed) nor at {src}")
            t0 = time.time()
            shutil.copy2(src, dst)
            log.info("OK mount copy %s (%.1f min)", name, (time.time() - t0) / 60)
    return dest


def copy_datasets_to_cache(mount_dirs, cache_dir: Path) -> None:
    """Plain-copy attached cache-dataset files to /tmp (sequential reads at
    the mount's ~26 MB/s — this is the only mount I/O training pays)."""
    for d in mount_dirs:
        for p in sorted(Path(d).iterdir()):
            if p.name.endswith(".zarr") or p.name == "X_test.nc":
                dst = cache_dir / p.name
                if not dst.exists():
                    log.info("copy %s", p.name)
                    t0 = time.time()
                    if p.is_dir():
                        shutil.copytree(p, dst)
                    else:
                        shutil.copy2(p, dst)
                    log.info("OK copy %s (%.1f min)", p.name, (time.time() - t0) / 60)
