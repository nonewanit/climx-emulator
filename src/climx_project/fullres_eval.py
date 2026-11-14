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


def evaluate_pass1(emulator, raw_test: Path, metadata, pred_path: Path,
                   time_chunk: int = 256) -> None:
    """Predict over the full 2015-2100 test forcings (chunk-wise numpy
    preprocessing) + reversal + fp16 final zarr."""
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
    if stage_path.exists():
        shutil.rmtree(stage_path)
    # transient store: no compression (faster appends; the final zarr is
    # the one that persists)
    from zarr.codecs import BloscCodec
    stage_enc = {"dtype": "float16",
                 "compressors": [BloscCodec(cname="zstd", clevel=0,
                                            shuffle="noshuffle")]}

    n = len(time)
    n_chunks = (n + time_chunk - 1) // time_chunk
    feature_order = list(emulator.feature_vars)
    import time as _time
    first = True
    for ci, t0 in enumerate(range(0, n, time_chunk)):
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
        enc = {v: stage_enc for v in ds_chunk.data_vars}
        if first:
            ds_chunk.to_zarr(stage_path, mode="w", encoding=enc)
            first = False
        else:
            ds_chunk.to_zarr(stage_path, mode="a", append_dim="time")
        del pred, ds_chunk
        log.info("  chunk %d: stage write %.1f s", ci + 1, _time.time() - t_w0)

    # Reversal (lazy, over the fp16 stage) + fp16 final write.
    y_pred_scaled = xr.open_zarr(stage_path, chunks={"time": 365})
    out = y_pred_scaled
    for var in list(out.data_vars):
        out[var] = out[var].astype(np.float32) * sc[var]["std"] + sc[var]["mean"]
    for var, info in log_info.items():
        if var in out:
            out[var] = np.minimum(out[var], 12)
            out[var] = (10 ** out[var]) - info["epsilon"]
    out = out.astype(np.float32)
    log.info("writing final predictions (fp16) to %s", pred_path)
    out.chunk({"time": 128, "lat": -1, "lon": -1}).to_zarr(
        pred_path, mode="w", encoding={v: stage_enc for v in out.data_vars})
    del out, y_pred_scaled
    shutil.rmtree(stage_path)
    log.info("pass 1 done; stage deleted")


def open_hist_from_mount(mount_dirs, cache_dir: Path) -> xr.Dataset:
    """Extract the hist-window stores ONE AT A TIME from the attached
    extras dataset and build an in-memory Dataset (peak ~2.4 GB disk)."""
    import zipfile
    needed = {"tas", "tasmax", "tasmin", "pr"}
    ds = xr.Dataset()
    for d in mount_dirs:
        for p in sorted(Path(d).glob("hist_*.zarrpkg")):
            var = p.name.split("_")[1]
            if var not in needed:
                continue
            store_dir = cache_dir / f"hist_{var}.zarr"
            if store_dir.exists():
                shutil.rmtree(store_dir)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(store_dir)
            if not (store_dir / "zarr.json").exists():
                (store_dir / "zarr.json").write_text(json.dumps(
                    {"zarr_format": 3, "node_type": "group", "attributes": {}}))
            da = xr.open_zarr(store_dir)[var]
            ds[var] = da.compute()
            shutil.rmtree(store_dir)
    if not ds.data_vars:
        raise FileNotFoundError("no hist_*.zarrpkg found in the attached datasets")
    return ds
