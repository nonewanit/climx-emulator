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
        # CHUNKED write (no full-array materialization): at x1 a full var is
        # 6.9 GB fp32 — the compute-then-write peaked ~10 GB per variable and
        # left the heap at the RAM limit, so even tiny later allocations OOM'd
        # the CPU session. The train file is LOCAL now; chunk-walk is fast.
        da.astype("float16").chunk(
            {d: (128 if d == "time" else -1) for d in da.dims}
        ).to_zarr(tmp, mode="w", encoding={name: encoding})
        tmp.rename(out)
        log.info("OK %s (%.1f min)", key, (time.time() - t0) / 60)
        if flush:
            flush()


def write_cache_y(full_dir: Path, cache_dir: Path, scenarios,
                  encoding: dict, scaling_json: Path,
                  skip: set | None = None, flush=None, step: int = 1) -> None:
    """y cache: SLAB-BASED direct netCDF4 reads (1,000-day contiguous
    hyperslabs — the mount's fast pattern — with ~220 MB transient memory).
    Transforms match the starter exactly: log10(x+eps) for pr/huss, then
    z-score with the frozen params."""
    from netCDF4 import Dataset as NCDataset

    import gc
    gc.collect()  # the X phase's full-var allocations leave a swollen heap
    sc = json.load(open(scaling_json))["scaling_params"]
    # SLAB must be a multiple of BOTH `step` (stride phase alignment) and
    # the 128-day zarr chunk grid (xarray rejects appends whose dask chunks
    # overlap multiple zarr chunks). 384 = lcm(3, 128) covers steps 1 and 3.
    SLAB = 384  # ~85 MB per read — RAM-safe even on the smallest CPU sessions
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
        # INCREMENTAL zarr appends: CPU sessions have small RAM (~4 GB) —
        # never hold the full variable in memory (~0.5 GB peak per slab).
        with NCDataset(str(full_dir / f"{FULL_PREFIX}train_{scen}.nc")) as nc:
            n = len(nc["time"])
            time_c = nc["time"][::step]
            lat = nc["lat"][:]
            lon = nc["lon"][:]
            # The vars are 4D (forcing_scenario, time, lat, lon): the slab
            # must slice the TIME axis, not axis 0. (Every previous y OOM was
            # this bug — [s0:s0+SLAB] sliced the scenario axis and pulled the
            # entire 6.9 GB variable per "slab".)
            var_dims = list(nc[var].dimensions)
            t_axis = var_dims.index("time")
            first = True
            idx = 0
            for s0 in range(0, n, SLAB):
                idx_slice = [slice(None)] * len(var_dims)
                idx_slice[t_axis] = slice(s0, s0 + SLAB)
                slab_arr = nc[var][tuple(idx_slice)]
                if len(slab_arr.shape) == 4 and slab_arr.shape[0] == 1:
                    slab_arr = slab_arr[0]  # squeeze the scenario dim
                s = slab_arr[::step].astype(np.float32)
                del slab_arr
                if var in ("pr", "huss"):
                    s = np.log10(s + 1e-6)
                s = (s - sc[var]["mean"]) / sc[var]["std"]
                s16 = s.astype("float16")
                del s
                da = xr.DataArray(s16, dims=("time", "lat", "lon"),
                                  coords={"time": time_c[idx:idx + len(s16)],
                                          "lat": lat, "lon": lon},
                                  name=var)
                ds_slab = xr.Dataset({var: da}).chunk(
                    {"time": 128, "lat": -1, "lon": -1})
                if first:
                    ds_slab.to_zarr(tmp, mode="w", encoding={var: encoding})
                    first = False
                else:
                    ds_slab.to_zarr(tmp, mode="a", append_dim="time")
                idx += len(s16)
                del s16, da, ds_slab
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
        # fp32 storage: psl (~100 kPa) OVERFLOWS fp16 (max 65,504) — the
        # fp16 cast produced inf/nan and an empty store (observed twice, with
        # 'overflow encountered in cast'). Chunked write, no full-array
        # materialization (CPU sessions have small RAM; the hist file is
        # LOCAL so chunk-walk speed is irrelevant).
        enc32 = dict(encoding)
        enc32["dtype"] = "float32"
        da.chunk({"time": 128, "lat": -1, "lon": -1}).to_zarr(
            tmp, mode="w", encoding={var: enc32})
        tmp.rename(out)
        gb = _dir_size(out) / 1e9
        log.info("OK hist %s (%.3f GB)", var, gb)
        if gb < 0.05:  # ~11k days of fp16 can't compress below ~50 MB
            log.error("hist %s store suspiciously small (%.3f GB) — REBUILD needed",
                      var, gb)
        if flush:
            flush()
    log.info("hist window written")


def open_hist_from_cache(cache_dir: Path) -> xr.Dataset:
    """Merge the cached hist_*.zarr window into a Dataset for evaluate()."""
    ds = xr.Dataset()
    for var in TARGETS:
        p = cache_dir / f"hist_{var}.zarr"
        if p.exists():
            try:
                ds[var] = xr.open_zarr(p)[var]
            except Exception as e:
                log.error("open %s failed: %s", p, e)
                log.error("store contents (first 30): %s",
                          sorted(str(f.relative_to(p)) for f in p.rglob("*"))[:30])
                raise
    if not ds.data_vars:
        raise FileNotFoundError(f"no hist_*.zarr under {cache_dir}")
    return ds


def upload_batches(cache_dir: Path, owner: str, manifest_path: Path,
                   batch_gb: float = 18.0,
                   slug_prefix: str = "climx-cache-b") -> dict:
    """Upload cache files in ~batch_gb batches as private Kaggle datasets
    (slug_prefix + idx), DELETING each batch from /tmp afterwards.

    This interleaves with the build (call it after each phase): the
    subsample-1 cache is ~170-200 GB — too big for one dataset and for one
    9 h session — and a session kill only loses the batch in flight.
    Parallel workers (one per scenario) pass distinct slug_prefixes
    (e.g. climx-cache-ssp126-b) and distinct manifest paths, so concurrent
    uploads never collide. The manifest ({slug: [stems]}) persists at
    manifest_path; earlier sessions' batches are skipped."""
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
        gb = _dir_size(p) / 1e9
        if batch and batch_size + gb > batch_gb:
            _upload_one(batch, owner, manifest_path, manifest, slug_prefix)
            batch, batch_size = [], 0.0
        batch.append(p)
        batch_size += gb
    if batch:
        _upload_one(batch, owner, manifest_path, manifest, slug_prefix)
    return manifest


def _dir_size(p: Path) -> float:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def _upload_one(batch: list[Path], owner: str, manifest_path: Path, manifest: dict,
                slug_prefix: str = "climx-cache-b", numbered: bool = True,
                extra_pkg_files: list[Path] | None = None,
                versioned: bool = False) -> None:
    idx = len(manifest) + 1
    slug = f"{owner}/{slug_prefix}{idx if numbered else ''}"
    stage = Path("/tmp/upload_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    import zipfile
    for p in batch:
        if p.is_dir():  # zarr stores are dirs; the CLI skips dirs. Package
            # them with a NON-.zip extension: Kaggle auto-extracts .zip files
            # on upload, which would shred the store into loose chunk files.
            zf = zipfile.ZipFile(stage / f"{p.name}.zarrpkg", "w",
                                 compression=zipfile.ZIP_STORED)
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(p))
            zf.close()
        else:
            shutil.copy2(p, stage / p.name)
    for p in (extra_pkg_files or []):
        shutil.copy2(p, stage / p.name)
    # CLI v2.2.5: metadata file is dataset-metadata.json (SINGULAR — verified
    # against the CLI's own error message); private is the default (there is
    # no --private flag).
    meta = {"title": f"{slug_prefix}{idx if numbered else ''}", "id": slug,
            "licenses": [{"name": "CC-BY-4.0"}]}
    with open(stage / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    gb = sum(_dir_size(p) for p in batch) / 1e9
    log.info("=== upload batch %d (%s): %.1f GB -> %s ===",
             idx, [p.name for p in batch], gb, slug)
    t0 = time.time()
    # create for NEW datasets; version for UPDATING an existing one (create
    # against an existing slug silently does nothing — observed: 0.1 min
    # "upload" that left the old files in place).
    cmd = (["kaggle", "datasets", "version", "-p", str(stage), "-m", "cache rebuild"]
           if versioned
           else ["kaggle", "datasets", "create", "-p", str(stage)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("datasets create failed (exit %d):\nSTDOUT: %s\nSTDERR: %s",
                  r.returncode, r.stdout, r.stderr)
        raise RuntimeError(f"kaggle datasets create failed: {r.stderr or r.stdout}")
    log.info("OK upload %s (%.1f min)", slug, (time.time() - t0) / 60)
    stems = [p.stem if p.suffix == ".zarr" else p.name for p in batch]
    manifest[slug] = sorted(set(manifest.get(slug, [])) | set(stems))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    for p in batch:
        if p.is_dir():
            shutil.rmtree(p)
        else:
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


def upload_cache(cache_dir: Path, owner: str, manifest_path: Path,
                 slug_prefix: str = "climx-cache",
                 include_pkg_from: Path | None = None,
                 versioned: bool = False) -> dict:
    """Upload ALL files in cache_dir as ONE private dataset (owner/slug_prefix).

    Per-scenario caches are ~60-80 GB compressed — one dataset each is
    clean and under the per-dataset allowance; the worker build (~3-4 h)
    fits the 9 h session cap, so end-of-build upload is fine.
    include_pkg_from: a dir whose .zarrpkg files are copied into the stage
    as-is (used by the y-rebuild recovery to re-upload the X packages with
    the corrected y packages as a new dataset version)."""
    manifest = {}
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
    uploaded = {f for files in manifest.values() for f in files}
    files = sorted([p for p in cache_dir.iterdir()
                    if (p.name.endswith(".zarr") or p.name == "X_test.nc")
                    and (p.stem if p.suffix == ".zarr" else p.name) not in uploaded
                    and not p.name.startswith(".tmp")],
                   key=lambda p: p.name)
    pkg_files = []
    if include_pkg_from:
        new_stems = {p.stem if p.suffix == ".zarr" else p.name for p in files}
        pkg_files = sorted(
            q for q in include_pkg_from.glob("*.zarrpkg")
            if q.name[:-8].rsplit(".zarr", 1)[0] not in new_stems)
    if not files:
        # nothing NEW to upload — skip even when a recover dir is present
        # (re-uploading a complete dataset as a redundant version is waste).
        log.info("nothing new to upload")
        return manifest
    _upload_one(files, owner, manifest_path, manifest,
                slug_prefix=slug_prefix, numbered=False,
                extra_pkg_files=pkg_files, versioned=versioned)
    return json.load(open(manifest_path))


def copy_datasets_to_cache(mount_dirs, cache_dir: Path) -> None:
    """Copy attached cache-dataset files to /tmp (sequential reads at the
    mount's ~26 MB/s — the only mount I/O training pays). Zarr stores
    arrive as <name>.zarr.zip (the upload zipped the dirs); unzip them."""
    import zipfile
    total_pkg = sum(p.stat().st_size for d in mount_dirs
                    for p in Path(d).iterdir()
                    if p.name.endswith(".zarrpkg") or p.name == "X_test.nc")
    df = subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True).stdout
    log.info("copy start: %.1f GB to unpack; %s", total_pkg / 1e9, df.strip().splitlines()[-1])
    for d in mount_dirs:
        for p in sorted(Path(d).iterdir()):
            if p.name.endswith(".zarrpkg"):
                dst = cache_dir / p.name[:-8]  # strip ".zarrpkg" -> <name>.zarr
                has_root = (dst / "zarr.json").exists() or (dst / ".zgroup").exists()
                if not has_root:
                    if dst.exists():
                        shutil.rmtree(dst)
                    log.info("unpack %s", p.name)
                    t0 = time.time()
                    with zipfile.ZipFile(p) as zf:
                        zf.extractall(dst)
                    # REPAIR: the chunked writers omit the ROOT group metadata
                    # (the per-array zarr.json files are complete) — add the
                    # minimal zarr v3 root so open_zarr finds the group.
                    if not (dst / "zarr.json").exists() and not (dst / ".zgroup").exists():
                        (dst / "zarr.json").write_text(json.dumps(
                            {"zarr_format": 3, "node_type": "group",
                             "attributes": {}}))
                        log.info("repaired missing root zarr.json for %s", p.name)
                    log.info("OK unpack %s (%.1f min)", p.name, (time.time() - t0) / 60)
            elif p.name.endswith(".zarr") or p.name == "X_test.nc":
                dst = cache_dir / p.name
                if not dst.exists():
                    log.info("copy %s", p.name)
                    t0 = time.time()
                    if p.is_dir():
                        shutil.copytree(p, dst)
                    else:
                        shutil.copy2(p, dst)
                    log.info("OK copy %s (%.1f min)", p.name, (time.time() - t0) / 60)
