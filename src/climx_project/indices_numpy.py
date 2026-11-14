"""Numpy streaming replacements for the three hist-dependent ClimX indices.

The starter's implementations (src/utils/indices_xclim.py) rechunk the time
dimension to ONE dask chunk (``chunk({'time': -1})``) so xclim's run-length
helpers are reliable — harmless on the lite 12x18 grid (~50 MB), but on the
full 192x288 grid a single chunk is 31,390 x 55,296 x 4-8 B = 7-14 GB per
array, and the WSDI graph OOM'd the Kaggle session. These versions do the
same math in numpy, streaming year by year (peak = one year + the
threshold):

- the percentile threshold replicates xclim-0.59's
  ``core.calendar.percentile_doy`` exactly (5-day centered rolling window
  with ``min_periods=1``, Hyndman-Fan type-8 NaN-aware quantile per
  day-of-year, alpha=beta=1/3), including the fp32 output dtype;
- WSDI/CSDI replicate the starter's per-year ``windowed_run_count``
  (sum of lengths of True-runs >= window; runs never cross year bounds);
- R95p replicates ``_pr_total_above_percentile`` (annual sum of pr on days
  above the day-of-year threshold; both sides in mm/day).

``climx_project.compat.apply()`` installs these as
``SELECTED_INDICES["WSDI"/"CSDI"/"R95p"]``, so both the full-res Kaggle
flow and the lite pipeline use them.
"""

import logging

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

KG_M2_S_TO_MM_DAY = 86400.0
DEFAULT_BASE_PERIOD = ("1961-01-01", "1990-12-31")


# ---------------------------------------------------------------- threshold

def _percentile_doy_numpy(vals: np.ndarray, doys: np.ndarray, per: float,
                          window: int = 5) -> np.ndarray:
    """Percentile per day-of-year — exact replica of xclim-0.59
    ``percentile_doy(alpha=beta=1/3)`` for a 365-day (noleap) calendar.

    vals: (T, lat, lon) daily values; doys: (T,) ints, 1..365.
    Returns (365, lat, lon) in vals.dtype.
    """
    if int(np.nanmax(doys)) > 365:
        raise NotImplementedError(
            "percentile_doy numpy replica supports only 365-day (noleap) "
            f"calendars; hist day-of-year max is {int(np.nanmax(doys))}")
    T, nlat, nlon = vals.shape
    q = per / 100.0
    offsets = np.arange(-(window // 2), window // 2 + 1)  # -2..+2 for window=5
    out = np.empty((365, nlat, nlon), dtype=np.float64)

    for d in range(365):
        rows = np.where(doys == d + 1)[0]
        if len(rows) == 0:
            out[d] = np.nan
            continue
        idx2 = rows[:, None] + offsets[None, :]      # (n_years, window)
        ok2 = (idx2 >= 0) & (idx2 < T)
        idx2c = np.where(ok2, idx2, 0)
        # only the per-doy window is upcast (the fp32 values are exact in
        # fp64, so sorting order and interpolation match xclim bit-for-bit)
        w = vals[idx2c].reshape(-1, nlat * nlon).astype(np.float64)
        w[~ok2.ravel()] = np.nan                     # rolling(min_periods=1) padding
        n_valid = w.shape[0] - np.isnan(w).sum(axis=0)
        w.sort(axis=0)                               # NaNs to the end
        # Hyndman-Fan type 8 virtual index: n*q + alpha + q*(1-alpha-beta) - 1
        h = n_valid * q + (1.0 / 3.0 + q * (1.0 - 2.0 / 3.0)) - 1.0
        few = n_valid < 2                            # xclim: yield the single
        h = np.where(few, 0.0, h)                    # value (or NaN) via nanmax
        above = h >= n_valid - 1                     # clamp: max value
        below = h < 0                                # clamp: min value
        prev_i = np.floor(h)
        gamma = h - prev_i
        prev_ii = np.where(above, -1, np.where(below, 0, prev_i)).astype(np.int64)
        next_ii = np.where(above, -1, np.where(below, 0, prev_i + 1)).astype(np.int64)
        prev_v = np.take_along_axis(w, prev_ii[None, :], axis=0)[0]
        next_v = np.take_along_axis(w, next_ii[None, :], axis=0)[0]
        interp = prev_v + gamma * (next_v - prev_v)
        interp = np.where(np.isnan(interp), np.nanmax(w, axis=0), interp)
        interp = np.where(few, np.nanmax(w, axis=0), interp)
        out[d] = interp.reshape(nlat, nlon)
    return out


def percentile_threshold_numpy(hist, var: str, per: float,
                               base_period, window: int = 5) -> np.ndarray:
    """(365, lat, lon) threshold from the hist dataset, in the hist dtype."""
    da = hist[var]
    if base_period:
        da = da.sel(time=slice(base_period[0], base_period[1]))
    # the lite hist carries a size-1 forcing_scenario dim; squeeze it away
    vals = np.squeeze(np.asarray(da.values))
    doys = np.asarray(da.time.dt.dayofyear.values, dtype=np.int64)
    return _percentile_doy_numpy(vals, doys, per, window)


# --------------------------------------------------------------- run counts

def _run_count_sum(mask: np.ndarray, window: int) -> np.ndarray:
    """Per column: sum of lengths of True-runs with length >= window
    (xclim's windowed_run_count over a single year — runs never cross
    year boundaries, matching the starter's per-year select)."""
    T, N = mask.shape
    prev = np.concatenate([np.zeros((1, N), dtype=bool), mask[:-1]], axis=0)
    starts = np.cumsum(mask & ~prev, axis=0)          # monotonic run ids
    labels = starts.astype(np.int64) * mask           # id per True cell only
    off = (np.arange(N, dtype=np.int64) * (T + 1))[None, :]
    counts = np.bincount((labels + off).ravel(),
                         minlength=N * (T + 1)).reshape(N, T + 1)[:, 1:]
    keep = counts >= window
    return (counts * keep).sum(axis=1)


def _precip_scale(da) -> float:
    units = (da.attrs.get("units", "") or "").lower()
    if units in ("mm/day", "mm d-1", "mm"):
        return 1.0
    return KG_M2_S_TO_MM_DAY


# ------------------------------------------------------------- index builds

def _annual_slices(data: xr.DataArray):
    years = np.unique(np.asarray(data.time.dt.year.values))
    lat = np.asarray(data.lat.values)
    lon = np.asarray(data.lon.values)
    return years, lat, lon


def wsdi(ds: xr.Dataset, historical_data_path=None,
         base_period=DEFAULT_BASE_PERIOD, freq: str = "YS",
         window: int = 5, **_) -> xr.DataArray:
    return _spell_index("WSDI", "tasmax", 90.0, ">", 6,
                        ds, historical_data_path, base_period, window)


def csdi(ds: xr.Dataset, historical_data_path=None,
         base_period=DEFAULT_BASE_PERIOD, freq: str = "YS",
         window: int = 5, **_) -> xr.DataArray:
    return _spell_index("CSDI", "tasmin", 10.0, "<", 6,
                        ds, historical_data_path, base_period, window)


def _spell_index(name, var, quantile, comparator, window_days,
                 ds, historical_data_path, base_period, window):
    if historical_data_path is None:
        raise ValueError(f"{name} requires 'historical_data_path'.")
    thr = percentile_threshold_numpy(historical_data_path, var, quantile,
                                     base_period, window)
    data = ds[var]
    years, lat, lon = _annual_slices(data)
    out = np.empty((len(years), len(lat), len(lon)), dtype=np.int64)
    for i, y in enumerate(years):
        sub = data.sel(time=slice(f"{y:04d}-01-01", f"{y:04d}-12-31"))
        # squeeze the lite data's size-1 forcing_scenario dim
        vals = np.squeeze(np.asarray(sub.values, dtype=np.float32))
        doys = np.asarray(sub.time.dt.dayofyear.values, dtype=np.int64) - 1
        mask = vals > thr[doys] if comparator == ">" else vals < thr[doys]
        out[i] = _run_count_sum(mask.reshape(len(mask), -1), window_days) \
            .reshape(len(lat), len(lon))
        del sub, vals, mask
    time = np.array([np.datetime64(f"{y:04d}-01-01", "ns") for y in years])
    return xr.DataArray(out, dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": lat, "lon": lon},
                        name=name)


def r95p(ds: xr.Dataset, historical_data_path=None,
         base_period=DEFAULT_BASE_PERIOD, freq: str = "YS",
         rechunk_time: int = 365, **_) -> xr.DataArray:
    if historical_data_path is None:
        raise ValueError("R95p requires 'historical_data_path'.")
    thr = percentile_threshold_numpy(historical_data_path, "pr", 95.0,
                                     base_period)
    thr = thr * _precip_scale(historical_data_path["pr"])   # mm/day
    data = ds["pr"]
    scale = _precip_scale(data)
    years, lat, lon = _annual_slices(data)
    out = np.empty((len(years), len(lat), len(lon)), dtype=np.float64)
    for i, y in enumerate(years):
        sub = data.sel(time=slice(f"{y:04d}-01-01", f"{y:04d}-12-31"))
        # squeeze the lite data's size-1 forcing_scenario dim
        vals = np.squeeze(np.asarray(sub.values, dtype=np.float32)) * scale
        doys = np.asarray(sub.time.dt.dayofyear.values, dtype=np.int64) - 1
        mask = vals > thr[doys]
        out[i] = np.where(mask, vals, 0.0).sum(axis=0)
        del sub, vals, mask
    time = np.array([np.datetime64(f"{y:04d}-01-01", "ns") for y in years])
    res = xr.DataArray(out, dims=("time", "lat", "lon"),
                       coords={"time": time, "lat": lat, "lon": lon},
                       name="R95p")
    res.attrs["units"] = "mm"
    return res
