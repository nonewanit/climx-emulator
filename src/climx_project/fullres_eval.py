"""Full-res evaluation helpers — the 86-year submission.

The submission parquet needs the 15 indices over the FULL 2015-2100 span
(86 yearly values x 55,296 cells — verified against sample_submission).
The preprocessed X_test for that span would be ~152 GB, so the test
forcings are transformed ON THE FLY per predict chunk IN NUMPY (the lazy
dask version recomputed the full-span monthly->daily interp per chunk -
5+ min each). The predictions stream to a fp16 zarr; the reversal + final
write happen in our own pass-1.
"""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .cacheio import zarr_fp16_encoding

log = logging.getLogger(__name__)

AEROSOLS = ["BC_AX", "BC_N", "SO2", "SO4_PR", "OM_NI"]
GHG = ["CO2_LBC", "N2O_LBC", "CH4_LBC", "CFC11eq_LBC", "CF2CL2_LBC"]


# Bumped after the fp16-clip + fp32-psl reversal fix: stores reversed by the
# OLD code contain infs (pr's 10^12 log-cap and psl's ~1e5 Pa overflowed the
# fp16 cast) and must be recomputed. Stores carry this marker as a sidecar.
REVERSAL_VERSION = "2"


def _reversal_marker(pred_path: Path) -> Path:
    return Path(str(pred_path) + ".reversal_version")


def evaluate_pass1(emulator, raw_test: Path, metadata, pred_path: Path,
                   time_chunk: int = 256) -> None:
    """Predict over the full 2015-2100 test forcings (chunk-wise numpy
    preprocessing) + reversal + fp16 final zarr."""
    if pred_path.exists():
        marker = _reversal_marker(pred_path)
        if marker.exists() and marker.read_text().strip() == REVERSAL_VERSION:
            log.info("final predictions exist (reversal v%s) — reusing "
                     "(skipping pass 1)", REVERSAL_VERSION)
            return
        log.warning("final predictions exist but lack reversal v%s — stale "
                    "store (pre-clip reversal, contains infs); recomputing",
                    REVERSAL_VERSION)
        shutil.rmtree(pred_path)
    sc = metadata["scaling_params"]["scaling_params"]
    log_info = metadata["scaling_params"].get("_log_transform_info", {})
    device = next(emulator.model.parameters()).device
    emulator.model.eval()

    ds = xr.open_dataset(raw_test, engine="h5netcdf",
                         chunks={"time": 512, "time_month": 24,
                                 "time_year": -1, "lat": -1, "lon": -1})
    time = ds["time"].values                       # (31389,) cftime
    lat = ds["lat"].values
    lon = ds["lon"].values
    def to_float(ts):
        d = ts - time[0]
        return np.asarray([x.days + x.seconds / 86400 for x in d])
    day_float = to_float(time)
    m_time = ds["time_month"].values               # (1032,)
    m_float = to_float(m_time)
    y_time = ds["time_year"].values
    y_float = to_float(y_time)

    # Read the small monthly/yearly arrays fully (fp64, ~2.3 GB total).
    monthly = {v: ds[v].isel(forcing_scenario=0).values for v in AEROSOLS if v in ds}
    yearly = {v: ds[v].isel(forcing_scenario=0).values for v in GHG if v in ds}

    def interp_to_days(src, src_float, day_float):
        # vectorized linear interp: bracket each day between the two
        # surrounding native time steps (searchsorted), no per-cell loop.
        lo = np.clip(np.searchsorted(src_float, day_float, side="right") - 1,
                     0, len(src_float) - 2)
        denom = src_float[lo + 1] - src_float[lo]
        w = np.where(denom > 0, (day_float - src_float[lo]) / denom, 0.0)
        w = w.reshape(-1, *([1] * (src.ndim - 1)))
        return src[lo] * (1 - w) + src[lo + 1] * w

    stage_path = Path("/tmp/pred_stage.zarr")
    reuse_stage = stage_path.exists() and (stage_path / "zarr.json").exists()
    if reuse_stage:
        log.info("stage exists — reusing it (skipping the predict loop)")
    else:
        if stage_path.exists():
            shutil.rmtree(stage_path)
    # transient store: no compression (faster appends; the final zarr is
    # the one that persists)
    from zarr.codecs import BloscCodec
    stage_enc = {"dtype": "float16",
                 "compressors": [BloscCodec(cname="zstd", clevel=0,
                                            shuffle="noshuffle")]}
    # psl is Pa (~1e5): every cell overflows fp16 after the reversal, and
    # a store full of infs poisons anything read back from it. Give psl a
    # fp32 store; the other six vars' anomalies fit fp16 comfortably.
    stage_enc_psl = dict(stage_enc, dtype="float32")

    n = len(time)
    n_chunks = (n + time_chunk - 1) // time_chunk
    feature_order = list(emulator.feature_vars)
    import time as _time
    first = True
    for ci, t0 in enumerate(range(0, n, time_chunk) if not reuse_stage else []):
        t1 = min(n, t0 + time_chunk)
        log.info("predict chunk %d/%d (days %d-%d)", ci + 1, n_chunks, t0, t1)
        t_pre0 = _time.time()
        chunk_days = day_float[t0:t1]
        X = np.empty((t1 - t0, 22, len(lat), len(lon)), dtype=np.float32)
        # aerosols: interp monthly -> chunk days (vectorized), log10, split
        for i, v in enumerate(AEROSOLS):
            d = interp_to_days(monthly[v], m_float, chunk_days) if v in monthly else np.zeros((t1 - t0, len(lat), len(lon)), np.float32)
            d = np.sign(d) * np.log10(np.abs(d) + 1e-6)
            X[:, feature_order.index(f"{v}_positive")] = np.where(d > 0, d, 0)
            X[:, feature_order.index(f"{v}_negative")] = np.where(d < 0, -d, 0)
            X[:, feature_order.index(f"{v}_zeros_mask")] = (d == 0).astype(np.float32)
        # GHGs: interp yearly -> chunk days, standardize
        for v in GHG:
            if v in yearly:
                d = interp_to_days(yearly[v], y_float, chunk_days)
                X[:, feature_order.index(v)] = ((d - sc[v]["mean"]) / sc[v]["std"]).reshape(-1, 1, 1)
        # seasonal
        doy = np.asarray([(t - time[0]).days % 365 for t in time[t0:t1]])
        phase = 2 * np.pi * doy / 365.25
        X[:, feature_order.index("sin_doy")] = np.sin(phase)[:, None, None]
        X[:, feature_order.index("cos_doy")] = np.cos(phase)[:, None, None]

        log.info("  chunk %d: preprocess %.1f s", ci + 1, _time.time() - t_pre0)
        t_fwd0 = _time.time()
        preds = []
        for s0 in range(0, len(X), 32):
            sub = torch.from_numpy(X[s0:s0 + 32]).to(device)
            with torch.no_grad():
                preds.append(emulator.model(sub).cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        del X, preds
        log.info("  chunk %d: forwards %.1f s", ci + 1, _time.time() - t_fwd0)
        t_w0 = _time.time()
        tcoord = time[t0:t1]
        ds_chunk = xr.Dataset(
            {v: (("time", "lat", "lon"), pred[:, i].astype(np.float16))
             for i, v in enumerate(emulator.target_vars)},
            coords={"time": tcoord, "lat": lat, "lon": lon},
        ).chunk({"time": 128, "lat": -1, "lon": -1})
        enc = {v: (stage_enc_psl if v == "psl" else stage_enc)
               for v in ds_chunk.data_vars}
        if first:
            ds_chunk.to_zarr(stage_path, mode="w", encoding=enc)
            first = False
        else:
            ds_chunk.to_zarr(stage_path, mode="a", append_dim="time")
        del pred, ds_chunk
        log.info("  chunk %d: stage write %.1f s", ci + 1, _time.time() - t_w0)

    # IN-PLACE reversal: overwrite the stage chunks with the physical values
    # (no second store — the session's effective write budget is ~45-50 GB
    # and the stage+final pair exceeded it, flipping /tmp read-only).
    import zarr
    g = zarr.open_group(str(stage_path), mode="a")
    for var in list(emulator.target_vars):
        a = g[var]
        n = a.shape[0]
        for t0 in range(0, n, 128):
            t1 = min(n, t0 + 128)
            vals = a[t0:t1].astype(np.float32) * sc[var]["std"] + sc[var]["mean"]
            if var in log_info:
                vals = np.minimum(vals, 12)
                vals = (10 ** vals) - log_info[var]["epsilon"]
            # Guard the fp16 stores: the log cap above means pr can reach
            # 10^12 mm/day in pathological cells — casting to fp16 overflows
            # to inf (RuntimeWarning), and infs poison the indices computed
            # from this store. Clip to the fp16 range; such cells are absurd
            # model outputs either way. psl's fp32 store skips the clip.
            if a.dtype == np.dtype("float16"):
                np.clip(vals, -65504.0, 65504.0, out=vals)
            a[t0:t1] = vals.astype(a.dtype)
    del g
    if pred_path.exists():
        shutil.rmtree(pred_path)
    stage_path.rename(pred_path)  # the stage IS the final predictions store
    _reversal_marker(pred_path).write_text(REVERSAL_VERSION)
    log.info("pass 1 done; stage became %s (reversal v%s)",
             pred_path, REVERSAL_VERSION)


def compute_indices_eagerly(indices_pred: dict) -> dict:
    """Materialize each index ONE AT A TIME.

    emulator.evaluate returns the indices as LAZY dask graphs over the
    prediction zarr + hist stores (its tqdm loop only builds graphs — the
    percentile indices especially). build_submission's ds.compute() then
    fuses all 15 graphs into one merged compute with shared chunk reads
    and xclim's float64 promotion — that merged compute OOM'd the Kaggle
    session. One at a time, the scheduler releases each graph before the
    next, bounding the peak to a single index.
    """
    import dask.array as da
    import gc

    out = {}
    for name, v in indices_pred.items():
        if isinstance(v.data, da.Array):
            log.info("computing index %s (lazy graph)", name)
            out[name] = v.compute()
            gc.collect()
        else:
            out[name] = v
    return out


def open_hist_from_mount(mount_dirs, cache_dir: Path) -> xr.Dataset:
    """Extract the hist-window stores ONE AT A TIME from the attached
    extras dataset and build an in-memory Dataset (peak ~2.4 GB disk)."""
    import zipfile
    needed = {"tas", "tasmax", "tasmin", "pr"}
    ds = xr.Dataset()
    for d in mount_dirs:
        for var in sorted(needed):
            # direct-name lookup: glob patterns were unreliable on the FUSE
            # mount (the files exist - verified - but glob found nothing)
            p = Path(d) / f"hist_{var}.zarr.zarrpkg"
            if not p.exists():
                continue
            store_dir = cache_dir / f"hist_{var}.zarr"
            if store_dir.exists():
                shutil.rmtree(store_dir)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(store_dir)
            if not (store_dir / "zarr.json").exists():
                (store_dir / "zarr.json").write_text(json.dumps(
                    {"zarr_format": 3, "node_type": "group", "attributes": {}}))
            # LAZY open (chunked): loading the window in memory (9.6 GB) on
            # top of the index computations OOM'd the kernel at index 7/15.
            # The stores stay on disk (~9.6 GB total, under the budget) and
            # the percentile computations stream chunk-wise.
            da = xr.open_zarr(store_dir, chunks={"time": 128, "lat": -1, "lon": -1})[var]
            ds[var] = da
    if not ds.data_vars:
        raise FileNotFoundError("no hist_*.zarrpkg found in the attached datasets")
    return ds
