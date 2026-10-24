"""Local validation that mirrors Kaggle scoring.

The official test scenario (SSP2-4.5) has no targets, so we hold out the last
20 years of a training scenario (SSP3-7.0, 2081-2100 — same convention as the
starter's NNBaseline.fit val split). Ground-truth indices are computed from
the raw daily fields with the official index code, both sides are upscaled
from the 12x18 lite grid to 192x288, and rnNSE.score() (the exact Kaggle
metric, AR6 mask embedded) produces the score.

Pitfall guarded here: rnNSE silently falls back to a non-regional scoring
path when the row count is not divisible by 192*288 — we hard-assert it.
"""

import inspect
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import paths

logger = logging.getLogger(__name__)

HOLDOUT_SCENARIO_FRAGMENT = "370"  # SSP3-7.0
HOLDOUT_START, HOLDOUT_END = "2081-01-01", "2100-12-31"
N_FULL_GRID = 192 * 288  # 55296


# ---------------------------------------------------------------- hold-out split

def _resolve_scenario_index(ds: xr.Dataset, dim: str = "forcing_scenario") -> int:
    if dim not in ds.dims:
        raise ValueError(f"Dataset has no {dim!r} dimension; dims={list(ds.dims)}")
    values = ds[dim].values
    for i, v in enumerate(values):
        if HOLDOUT_SCENARIO_FRAGMENT in str(v):
            return i
    raise ValueError(
        f"No scenario matching {HOLDOUT_SCENARIO_FRAGMENT!r} in {dim} values: "
        f"{[str(v) for v in values]}"
    )


def _drop_aux_time_dims(ds: xr.Dataset) -> xr.Dataset:
    """Mirror preprocess_train's defensive cleanup of time_month/time_year."""
    ds = ds.drop_vars(["time_month", "time_year"], errors="ignore")
    if "time_month" in ds.dims:
        ds = ds.isel(time_month=0, drop=True)
    if "time_year" in ds.dims:
        ds = ds.isel(time_year=0, drop=True)
    return ds


def make_holdout_split(X_train: xr.Dataset, y_train: xr.Dataset, dataset):
    """Return (X_hold, y_raw) for SSP3-7.0, 2081-2100.

    X_hold: preprocessed (scaled anomaly) forcing inputs, dims (time, lat, lon).
    y_raw:  raw physical daily target fields from the training dataset — the
            ground truth against which holdout indices are computed.
    """
    scen_idx = _resolve_scenario_index(X_train)
    X_hold = X_train.isel(forcing_scenario=scen_idx).sel(time=slice(HOLDOUT_START, HOLDOUT_END))
    X_hold = _drop_aux_time_dims(X_hold)

    raw = dataset.train
    raw_idx = _resolve_scenario_index(raw)
    y_raw = raw.isel(forcing_scenario=raw_idx).sel(time=slice(HOLDOUT_START, HOLDOUT_END))
    y_raw = _drop_aux_time_dims(y_raw)
    y_raw = y_raw[paths.TARGET_VARIABLES]

    assert X_hold.sizes["time"] == y_raw.sizes["time"], (
        f"Holdout time mismatch: X={X_hold.sizes['time']}, y={y_raw.sizes['time']}"
    )
    return X_hold, y_raw


# ---------------------------------------------------------------- index helpers

def _call_index_function(func, ds: xr.Dataset, index_name: str, historical_data_path):
    """Replicate the starter's kwargs injection for percentile-based indices."""
    try:
        params = inspect.signature(func).parameters
    except (ValueError, TypeError):
        params = {}
    kwargs = {}
    if "historical_data_path" in params:
        if historical_data_path is None:
            raise ValueError(
                f"Index '{index_name}' requires 'historical_data_path' but none was provided."
            )
        kwargs["historical_data_path"] = historical_data_path
    return func(ds, **kwargs)


def compute_gt_indices(y_raw_daily: xr.Dataset, historical_data_path) -> dict[str, xr.DataArray]:
    """Official index code applied to RAW physical daily fields (no preprocessing reversal)."""
    from src import indices_xclim as I

    gt = {}
    for name, func in I.SELECTED_INDICES.items():
        logger.info("GT index: %s", name)
        gt[name] = _call_index_function(func, y_raw_daily, name, historical_data_path)
    return gt


def _format_and_dataset(indices: dict[str, xr.DataArray], model_type: str) -> xr.Dataset:
    """format_indices (calendar unify + TXx/TNn/Rx5day annualization), drop any
    forcing_scenario dim (the starter only does this when saving), to Dataset.

    Also asserts all 15 indices end up on ONE shared annual time axis — xclim's
    yearly indices and the resampled monthly ones must agree (otherwise the
    submission would silently carry NaNs after to_array alignment).
    """
    from src.utils.indices_utils import format_indices

    formatted = format_indices(indices, model_type, paths.DATA_VERSION, save_indices=False)
    for k, v in formatted.items():
        if "forcing_scenario" in v.dims:
            formatted[k] = v.isel(forcing_scenario=0, drop=True)

    ds = xr.Dataset(formatted)
    ref = None
    for k in ds.data_vars:
        t = np.asarray(ds[k]["time"].values)
        if ref is None:
            ref = t
        elif t.shape != ref.shape or not np.array_equal(t, ref):
            raise RuntimeError(
                f"{model_type}: index {k!r} time axis ({t.shape}) differs from the "
                f"others ({ref.shape}) — annualization produced inconsistent calendars."
            )
    return ds


def upscale(indices_ds: xr.Dataset) -> xr.Dataset:
    """12x18 -> 192x288 nearest-neighbour with edge extrapolation.

    fill_value='extrapolate' is mandatory: the full grid extends beyond the
    coarse grid bounds and xarray would otherwise inject edge NaNs, which the
    scorer rejects (and the parquet writer silently zero-fills).
    """
    from src.utils.original_data_values import LAT, LON

    lat = np.asarray(LAT, dtype=float)
    lon = np.asarray(LON, dtype=float)
    return (
        indices_ds.sortby("lat")
        .sortby("lon")
        .interp(lat=lat, lon=lon, method="nearest", kwargs={"fill_value": "extrapolate"})
    )


def to_df(indices_ds: xr.Dataset) -> pd.DataFrame:
    import src.utils.kaggle_submission as ks

    return ks.indices_xr_to_kaggle_dataframe(indices_ds)


def score_local(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Official rnNSE scorer. Hard-guards the regional scoring path."""
    import rnNSE

    if len(gt_df) != len(pred_df):
        raise ValueError(f"Row count mismatch: GT={len(gt_df)}, pred={len(pred_df)}")
    if len(gt_df) % N_FULL_GRID != 0:
        raise ValueError(
            f"{len(gt_df)} rows is not divisible by {N_FULL_GRID}; rnNSE would silently "
            "fall back to non-regional scoring."
        )
    return float(rnNSE.score(gt_df, pred_df, "row_id"))


# ---------------------------------------------------------------- full flows

def run_holdout_validation(emulator, model_name: str, X_hold: xr.Dataset,
                           y_raw: xr.Dataset, metadata: dict,
                           historical_data_path) -> float:
    """Predict on the holdout, compute GT indices, score with the official metric."""
    logger.info("Holdout validation on SSP3-7.0 %s..%s ...", HOLDOUT_START, HOLDOUT_END)

    y_pred_hold, indices_pred = emulator.evaluate(
        X_hold,
        metadata,
        results_path=paths.INDICES_DIR / f"{model_name}_holdout_evaluation.json",
        predictions_path=paths.PREDICTIONS_DIR / f"{model_name}_holdout_predictions.zarr",
        historical_data_path=historical_data_path,
        TARGET_VARIABLES=paths.TARGET_VARIABLES,
        LOAD_PREDICTIONS=False,
        compute_indices=True,
    )

    gt_indices = compute_gt_indices(y_raw, historical_data_path)

    ds_gt = _format_and_dataset(gt_indices, "ground_truth")
    ds_pred = _format_and_dataset(indices_pred, model_name)

    # Parity check: both sides must end up with the same annual time axis.
    ds_gt, ds_pred = xr.align(ds_gt, ds_pred, join="inner")
    n_t = ds_gt.sizes["time"]
    if n_t != ds_pred.sizes["time"] or n_t < 1:
        raise RuntimeError(f"Formatted time axes disagree: GT={ds_gt.sizes['time']}, pred={ds_pred.sizes['time']}")

    # Keep the formatted/upscaled copies for inspection.
    ds_gt_ups = upscale(ds_gt)
    ds_pred_ups = upscale(ds_pred)
    ds_gt_ups.to_netcdf(paths.INDICES_DIR / f"{model_name}_holdout_gt_indices.nc")
    ds_pred_ups.to_netcdf(paths.INDICES_DIR / f"{model_name}_holdout_pred_indices.nc")

    score = score_local(to_df(ds_gt_ups), to_df(ds_pred_ups))
    logger.info("Local nNSE (%s, %d years): %.4f", model_name, n_t, score)
    return score


def self_check_climatology(emulator, X_hold: xr.Dataset, metadata: dict,
                           historical_data_path) -> float:
    """Wiring test: score the climatology emulator against its own predictions.

    Expect ~1.0 — this isolates all plumbing (formatting, upscaling, DataFrame
    construction, row alignment) from model quality.
    """
    y_pred, indices_pred = emulator.evaluate(
        X_hold,
        metadata,
        results_path=paths.INDICES_DIR / "climatology_selfcheck_evaluation.json",
        predictions_path=paths.PREDICTIONS_DIR / "climatology_selfcheck_predictions.zarr",
        historical_data_path=historical_data_path,
        TARGET_VARIABLES=paths.TARGET_VARIABLES,
        LOAD_PREDICTIONS=False,
        compute_indices=True,
    )
    gt = {k: v.copy(deep=True) for k, v in indices_pred.items()}
    ds_gt = _format_and_dataset(gt, "selfcheck_gt")
    ds_pred = _format_and_dataset(indices_pred, "selfcheck_pred")
    ds_gt, ds_pred = xr.align(ds_gt, ds_pred, join="inner")
    score = score_local(to_df(upscale(ds_gt)), to_df(upscale(ds_pred)))
    logger.info("Self-check nNSE (expect ~1.0): %.6f", score)
    return score


def append_score(model: str, local_score: float, parquet: Path | None,
                 kaggle_score: float | None = None) -> None:
    """Append a row to results/scores.md (the tracked score log)."""
    paths.SCORES_MD.parent.mkdir(parents=True, exist_ok=True)
    header = "| date | model | local nNSE (SSP370 holdout) | parquet | kaggle LB | commit |"
    sep = "|---|---|---|---|---|---|"
    row = (
        f"| {date.today().isoformat()} | {model} | {local_score:.4f} | "
        f"{parquet.name if parquet else '-'} | "
        f"{f'{kaggle_score:.4f}' if kaggle_score is not None else '-'} | "
        f"{_git_short_hash()} |"
    )
    if not paths.SCORES_MD.exists():
        paths.SCORES_MD.write_text(header + "\n" + sep + "\n")
    with open(paths.SCORES_MD, "a") as f:
        f.write(row + "\n")


def _git_short_hash() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "-"
