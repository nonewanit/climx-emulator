"""Compatibility shims for third-party library bugs found on this stack.

1. xarray's `_strftime_through_cftimeindex`: with pandas 2.2 + pyarrow,
   `CFTimeIndex.strftime()` returns an `ArrowStringArray`, whose `.reshape()`
   is not implemented — so any `.dt.strftime(...)` on a cftime time axis
   raises `NotImplementedError`. This breaks xclim's percentile-based indices
   (WSDI/CSDI/R95p), which call `build_climatology_bounds()` on the
   historical (noleap, cftime) time axis. Fix: coerce the array to plain
   numpy before reshaping. The datetime64 path (`_strftime_through_series`)
   is patched the same way preemptively.

2. Starter-kit stationarization reversal (see `_patch_stationarize_reverse`).

3. Hist-dependent index replacements (see `_patch_hist_indices`).
"""

import logging

import numpy as np
import xarray.core.accessor_dt as _xr_dt

logger = logging.getLogger(__name__)


def _as_numpy(arr):
    """numpy array from any pandas extension-array (ArrowStringArray etc.)."""
    return arr if isinstance(arr, np.ndarray) else np.asarray(arr.tolist())


def _strftime_through_cftimeindex(values, date_format: str):
    from xarray.coding.cftimeindex import CFTimeIndex
    from xarray.core import duck_array_ops

    values_as_cftimeindex = CFTimeIndex(duck_array_ops.ravel(values))
    field_values = values_as_cftimeindex.strftime(date_format)
    return _as_numpy(field_values.values).reshape(values.shape)


def _strftime_through_series(values, date_format: str):
    import pandas as pd
    from xarray.core import duck_array_ops

    values_as_series = pd.Series(duck_array_ops.ravel(values), copy=False)
    strs = values_as_series.dt.strftime(date_format)
    return _as_numpy(strs.values).reshape(values.shape)


def _patch_stationarize_reverse() -> None:
    """Fix the starter's broken stationarization reversal.

    `_stationarize_single_scenario` (third_party/climx) broadcasts the RAW
    monthly climatology into a new 'month' dim in reverse mode; evaluate()
    then collapses it with `isel(month=0)` — January's climatology is added
    to EVERY day of the year (annual-mean bias ~2 K; summer days ~15-20 K
    too cold). The daily-interpolation code that was clearly intended is
    present but commented out. This patch restores that intent: in reverse
    mode, interpolate the monthly climatology to the data's daily time axis
    first — the exact inverse of what preprocess_train does in forward mode.
    """
    import src.data_preprocessing.preprocessing_utils as ppu

    original = ppu._stationarize_single_scenario

    def patched(data, climatology, reverse=False, stationarization_mode="monthly",
                daily_climatology=None):
        if reverse and "monthly" in stationarization_mode:
            daily = ppu.monthly_climatology_to_daily(climatology, data.time)
            return data + daily
        return original(
            data, climatology, reverse=reverse,
            stationarization_mode=stationarization_mode,
            daily_climatology=daily_climatology,
        )

    ppu._stationarize_single_scenario = patched
    logger.info("Applied stationarize(reverse) daily-climatology patch")


def _patch_compute_scaling_params() -> None:
    """Make scaling-param computation a single-pass reduction per variable.

    The starter's compute_scaling_params leaves the data in ~30-day chunks
    (~200 tasks per variable over the 6000-day sample) and triggers a
    separate compute per variable — observed ~42 min for the full set, and
    the monthly-mode variant re-runs the whole anomaly graph per variable.
    Rechunking each variable to ONE chunk collapses it to 1 task per
    variable (identical values; skipna semantics preserved), ~1-2 min
    total on local data.
    """
    import src.data_preprocessing.preprocessing as pp
    import src.data_preprocessing.preprocessing_utils as ppu

    def patched(ds, target_vars, forcing_vars, log_transform_info,
                aerosol_threshold=0.1, std_epsilon=1e-9):
        all_vars = target_vars + forcing_vars
        ds_subset = ds[all_vars].chunk({"time": -1})
        means = ds_subset.mean(skipna=True).compute()
        stds = ds_subset.std(skipna=True).compute()
        scaling_params = {}
        for var in all_vars:
            scaling_params[var] = {
                "mean": float(means[var].values.item()),
                "std": float(stds[var].values.item()),
            }
        return scaling_params

    pp.compute_scaling_params = patched       # the name preprocess_train calls
    ppu.compute_scaling_params = patched      # any direct importers
    logger.info("Applied single-pass compute_scaling_params patch")


def _patch_hist_indices() -> None:
    """Replace the hist-dependent indices (WSDI/CSDI/R95p) with the numpy
    streaming versions from indices_numpy.

    The starter's dask implementations rechunk time to ONE chunk so xclim's
    run-length helpers work; on the lite 12x18 grid that is ~50 MB, but on
    the full 192x288 grid a single chunk is 7-14 GB per array and the WSDI
    graph OOM'd the Kaggle session. The numpy versions stream year by year
    with the exact same math (see indices_numpy module docstring).
    """
    import src.indices_xclim as I
    from . import indices_numpy

    _ORIGINAL_HIST_INDICES.update(
        {k: I.SELECTED_INDICES[k] for k in ("WSDI", "CSDI", "R95p")})
    I.SELECTED_INDICES["WSDI"] = indices_numpy.wsdi
    I.SELECTED_INDICES["CSDI"] = indices_numpy.csdi
    I.SELECTED_INDICES["R95p"] = indices_numpy.r95p
    logger.info("Applied numpy WSDI/CSDI/R95p replacements")


_ORIGINAL_HIST_INDICES: dict = {}


def apply() -> None:
    _xr_dt._strftime_through_cftimeindex = _strftime_through_cftimeindex
    _xr_dt._strftime_through_series = _strftime_through_series
    _patch_stationarize_reverse()
    _patch_compute_scaling_params()
    _patch_hist_indices()
    logger.info("Applied xarray strftime/ArrowStringArray compatibility patch")
