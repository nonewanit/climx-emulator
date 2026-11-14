"""Full-res evaluation helpers — the 86-year submission.

The submission parquet needs the 15 indices over the FULL 2015-2100 span
(86 yearly values x 55,296 cells — verified against sample_submission).
The preprocessed X_test for that span would be ~152 GB, so the test
forcings are transformed ON THE FLY (lazy dask, per predict chunk), the
predictions stream to a fp16 zarr, and the reversal + final write happen
in our own pass-1 (the starter's in-memory flow peaked at ~25 GB RAM).
"""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from .cacheio import zarr_fp16_encoding

log = logging.getLogger(__name__)

AEROSOLS = ["BC_AX", "BC_N", "SO2", "SO4_PR", "OM_NI"]
GHG = ["CO2_LBC", "N2O_LBC", "CH4_LBC", "CFC11eq_LBC", "CF2CL2_LBC"]


def build_test_forcings_lazy(raw_path: Path, scaling_json: Path) -> xr.Dataset:
    """Lazy daily test forcings over the FULL 2015-2100 span, preprocessed
    exactly like preprocess_test's none-mode (frozen params)."""
    from src.data_preprocessing.preprocessing_utils import upsample_forcing_to_daily

    sc = json.load(open(scaling_json))["scaling_params"]
    ds = xr.open_dataset(raw_path, engine="h5netcdf",
                         chunks={"time": 512, "time_month": 24,
                                 "time_year": -1, "lat": -1, "lon": -1})
    yearly = [v for v in GHG if v in ds]
    monthly = [v for v in AEROSOLS if v in ds]
    parts = []
    if yearly:
        parts.append(upsample_forcing_to_daily(ds, yearly, "time_year"))
    if monthly:
        parts.append(upsample_forcing_to_daily(ds, monthly, "time_month"))
    X = xr.merge(parts)
    X = X.drop_vars(["time_month", "time_year"], errors="ignore")
    for dim in ("time_month", "time_year"):
        if dim in X.dims:
            X = X.isel({dim: 0}, drop=True)

    # log10 for aerosols (sign * log10(|x| + eps)) then split
    for var in AEROSOLS:
        X[var] = np.sign(X[var]) * np.log10(np.abs(X[var]) + 1e-6)
        X[f"{var}_positive"] = X[var].where(X[var] > 0, 0)
        X[f"{var}_negative"] = -(X[var].where(X[var] < 0, 0))
        X[f"{var}_zeros_mask"] = (X[var] == 0).astype(np.float32)
        X = X.drop_vars(var)

    # standardize the GHGs with the frozen params
    for var in GHG:
        if var in X:
            X[var] = (X[var] - sc[var]["mean"]) / sc[var]["std"]

    # seasonal features
    doy = 2 * np.pi * X.time.dt.dayofyear / 365.25
    X["sin_doy"] = np.sin(doy).expand_dims({"lat": X.lat, "lon": X.lon}).transpose("time", "lat", "lon")
    X["cos_doy"] = np.cos(doy).expand_dims({"lat": X.lat, "lon": X.lon}).transpose("time", "lat", "lon")
    X = X.astype(np.float32)
    return X


def evaluate_pass1(emulator, X_test, metadata, pred_path: Path) -> None:
    """Predict (streaming, fp16) + reversal + fp16 final zarr — RAM- and
    disk-bounded (the starter's flow peaked at ~25 GB RAM / 94 GB disk)."""
    y_pred_scaled = emulator.predict(X_test)  # lazy fp16 stage at /tmp/pred_stage.zarr
    sc = metadata["scaling_params"]["scaling_params"]
    log_info = metadata["scaling_params"].get("_log_transform_info", {})

    out = y_pred_scaled
    for var in list(out.data_vars):
        out[var] = out[var].astype(np.float32) * sc[var]["std"] + sc[var]["mean"]
    for var, info in log_info.items():
        if var in out:
            out[var] = np.minimum(out[var], 12)
            out[var] = (10 ** out[var]) - info["epsilon"]
    out = out.astype(np.float32)

    enc = zarr_fp16_encoding()
    log.info("writing final predictions (fp16) to %s", pred_path)
    out.chunk({"time": 128, "lat": -1, "lon": -1}).to_zarr(
        pred_path, mode="w", encoding={v: enc for v in out.data_vars})
    del out, y_pred_scaled
    stage = Path("/tmp/pred_stage.zarr")
    if stage.exists():
        shutil.rmtree(stage)
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
