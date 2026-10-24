"""Compatibility shims for third-party library bugs found on this stack.

1. xarray's `_strftime_through_cftimeindex`: with pandas 2.2 + pyarrow,
   `CFTimeIndex.strftime()` returns an `ArrowStringArray`, whose `.reshape()`
   is not implemented — so any `.dt.strftime(...)` on a cftime time axis
   raises `NotImplementedError`. This breaks xclim's percentile-based indices
   (WSDI/CSDI/R95p), which call `build_climatology_bounds()` on the
   historical (noleap, cftime) time axis. Fix: coerce the array to plain
   numpy before reshaping. The datetime64 path (`_strftime_through_series`)
   is patched the same way preemptively.
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


def apply() -> None:
    _xr_dt._strftime_through_cftimeindex = _strftime_through_cftimeindex
    _xr_dt._strftime_through_series = _strftime_through_series
    logger.info("Applied xarray strftime/ArrowStringArray compatibility patch")
