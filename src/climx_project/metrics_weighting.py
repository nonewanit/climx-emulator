"""Scorer-derived loss weights (metric-consistent weighting).

Mirrors the implicit weighting of rnNSE.score (third_party/climx/rnNSE.py) for
a daily-field error at cell (i, j) of variable v:

- only AR6 land-region cells count — ocean weight is 0 (the scorer's mask);
- per-cell TEMPORAL variance normalization 1/Var (the R² denominator — a
  low-variance cell's error is amplified by 1/Var);
- area weighting cos(lat) inside a region, and every region averaged EQUALLY
  regardless of size (w ∝ cosφ / Σ_region cosφ);
- the 15 indices are averaged equally (1/15 each), mapped to the daily
  variables they consume (see INDEX_VARIABLES).

Output: a fixed (n_vars, H, W) fp32 tensor of per-cell, per-variable weights
for the weighted-MSE loss in SCALED-ANOMALY space. Computed once from the
training-scenario GT (train slice 2015-2080 — NEVER the holdout), clamped and
mean-normalized so the loss keeps a comparable scale. No learned weights, no
tuning knobs: the only constants are the variance floor (relative, by
percentile-free formula) and the clamp percentile, both safety mechanisms
against the scorer's 1/Var explosion, not model choices.

Day-level weights (extreme-day sensitivity) are deliberately NOT included:
V2 adds active-day masks (which days actually move each index) as an
importance-sampling weight over training days.
"""

import logging
from collections import Counter

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# The 15 leaderboard indices -> the daily variable each one is computed from
# (third_party/climx/src/indices_xclim.py SELECTED_INDICES).
INDEX_VARIABLES = {
    "FD": "tasmin", "SU": "tasmax", "ID": "tasmax", "TR": "tasmin",
    "GSL": "tas", "TXx": "tasmax", "TNn": "tasmin",
    "WSDI": "tasmax", "CSDI": "tasmin",
    "Rx5day": "pr", "CDD": "pr", "CWD": "pr",
    "R95p": "pr", "SDII": "pr", "R10mm": "pr",
}

# huss/psl/sfcWind are part of the task but appear in ZERO indices. A small
# floor (relative to the mean used-variable weight) keeps them trainable.
UNUSED_VARIABLE_FLOOR = 0.2

# 1/Var safety: cap cell weights at this quantile of the positive weights so
# the lowest-variance cells cannot destabilize training (the scorer itself
# drops Var < 1e-6 cells entirely — our floor+clamp is the loss-side analog).
CLAMP_QUANTILE = 0.95


def decode_ar6_mask() -> np.ndarray:
    """The scorer's embedded AR6 land-region mask -> (192, 288) int16.

    Region code >= 0 = AR6 land region, -1 = ocean / outside AR6 land.
    """
    import rnNSE as _rn  # third_party/climx root, on sys.path via climx_project

    flat = np.empty(_rn._NLAT * _rn._NLON, dtype=np.int16)
    i = 0
    for code, run in _rn._MASK_RLE:
        flat[i:i + run] = code
        i += run
    assert i == len(flat), f"RLE decoded {i} cells, expected {len(flat)}"
    return flat.reshape(_rn._NLAT, _rn._NLON)


def coarsen_mask(region_mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Coarsen the 192x288 mask to (out_h, out_w) by per-block MODE region code.

    Requires exact divisibility (lite 12x18 = 16x16 blocks). Mode (not
    majority-of-land) mirrors how the lite prediction is nearest-neighbour
    upscaled to the full grid: each lite cell stands in for its whole block,
    including its ocean cells.
    """
    H, W = region_mask.shape
    if (out_h, out_w) == (H, W):
        return region_mask
    if H % out_h or W % out_w:
        raise ValueError(f"192x288 mask not divisible by ({out_h}, {out_w})")
    blocks = region_mask.reshape(out_h, H // out_h, out_w, W // out_w)
    out = np.empty((out_h, out_w), dtype=region_mask.dtype)
    for i in range(out_h):
        for j in range(out_w):
            vals = blocks[i, :, j, :].ravel()
            codes, counts = np.unique(vals, return_counts=True)
            out[i, j] = codes[int(np.argmax(counts))]
    return out


def compute_metric_weights(var_cell: np.ndarray, lats: np.ndarray,
                           region_mask: np.ndarray, target_vars: list,
                           index_variables: dict | None = None) -> np.ndarray:
    """Weight tensor from per-cell variances + the scorer's aggregation.

    var_cell: (V, H, W) per-variable temporal variance of the SCALED-ANOMALY
        training fields (loss space — the relative cell ordering matches the
        scorer's raw-space variance up to the log-transform of pr).
    lats: (H,) latitudes in degrees; region_mask: (H, W) int (>=0 land, -1 ocean).
    Returns fp32 (V, H, W), mean over ALL cells and vars == 1.
    """
    V, H, W = var_cell.shape
    assert region_mask.shape == (H, W) and len(lats) == H and V == len(target_vars)

    # Per-variable weight: equal index weight (1/15) mapped to the daily
    # variables each index consumes; floor for the variables no index uses.
    index_variables = index_variables or INDEX_VARIABLES
    counts = Counter(index_variables.values())
    n_idx = max(1, len(index_variables))
    w_var = np.array([counts.get(v, 0) / n_idx for v in target_vars],
                     dtype=np.float64)
    used_mean = w_var[w_var > 0].mean()
    w_var[w_var == 0] = UNUSED_VARIABLE_FLOOR * used_mean

    land = region_mask >= 0
    land_idx = np.where(land)
    cos_w = np.cos(np.deg2rad(lats)).astype(np.float64)

    # Region size normalization: Σ cosφ over the region's cells (regions are
    # averaged equally by the scorer, so cells in big regions count less).
    z_region = np.ones((H, W), dtype=np.float64)
    for rc in np.unique(region_mask[land]):
        in_r = region_mask == rc
        z_region[in_r] = np.maximum((cos_w[:, None] * in_r).sum(), 1e-12)

    # Work on land-flattened arrays (clearer than masked broadcasting).
    cos_l = cos_w[land_idx[0]]                          # (n_land,)
    var_l = var_cell[:, land_idx[0], land_idx[1]]       # (V, n_land)
    zr_l = z_region[land_idx[0], land_idx[1]]           # (n_land,)
    # 1/Var with a per-variable relative floor (safety, see module docstring).
    eps_v = np.array([max(1e-6, 0.01 * var_l[v].mean()) for v in range(V)])

    w_l = cos_l[None, :] / (var_l + eps_v[:, None]) / zr_l[None, :]
    w_cell = np.zeros((V, H, W), dtype=np.float64)
    w_cell[:, land] = w_l
    w = w_cell * w_var[:, None, None]

    # Clamp the 1/Var explosion at a fixed quantile, then mean-normalize over
    # ALL cells (ocean zeros included) so the loss scale stays comparable to
    # the previous uniform-cell loss.
    pos = w[:, land]
    cap = float(np.percentile(pos, 100 * CLAMP_QUANTILE))
    w = np.minimum(w, cap)
    w /= max(w.mean(), 1e-12)

    logger.info(
        "metric weights: %d/%d land cells (%.1f%%), region count %d, "
        "weight p50/p95/max %.3f/%.3f/%.3f, per-var means %s",
        int(land.sum()), H * W, 100 * land.mean(),
        len(np.unique(region_mask[land])),
        float(np.percentile(pos, 50)), float(np.percentile(pos, 95)), pos.max(),
        np.array2string(w.mean(axis=(1, 2)), precision=3))
    return w.astype(np.float32)


def compute_metric_weights_for_fit(y_train_ds: xr.Dataset, target_vars: list,
                                   train_slice=("2015-01-01", "2080-12-31")
                                   ) -> np.ndarray:
    """End-to-end: scorer-derived weight tensor from the preprocessed y_train.

    Variance is computed per scenario over the TRAIN slice (never the
    holdout 2081-2100) and averaged over scenarios; the mask is decoded from
    the vendored scorer and coarsened to y_train's grid.
    """
    y = y_train_ds[target_vars].sel(time=slice(*train_slice))
    H, W = y.sizes["lat"], y.sizes["lon"]
    mask = coarsen_mask(decode_ar6_mask(), H, W)

    # Dask-backed per-cell temporal variance, one pass per variable.
    var_ds = y.var(dim="time").mean("forcing_scenario").compute()
    var_cell = var_ds.to_array(dim="feature").values.astype(np.float64)

    return compute_metric_weights(var_cell, np.asarray(y["lat"], dtype=float),
                                  mask, target_vars)
