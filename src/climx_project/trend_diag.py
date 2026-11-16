"""Trend diagnostic: does the model capture the forced warming?

The scorer's per-cell R2 = 1 - MSE/Var over the 86-year test period is won
or lost on the FORCED WARMING TREND in the temperature indices: a model
whose annual tas/tasmax anomaly tracks the ground truth gets R2 ~0.8+ on
those indices (nNSE ~ +0.7); a model that misses the trend is dragged
toward -1. This script plots predicted vs GT annual-mean tas over the
SSP370 holdout (2081-2100, land cells) and prints the linear trend slopes
(K/decade), the mean bias, and the correlation — one look tells you whether
to invest in training longer or fix the forcing encoding.

Usage (Kaggle notebook cell):
    import glob
    best = sorted(glob.glob("results/models/unet/best-*.ckpt"))[-1]
    !python -m climx_project.trend_diag unet --checkpoint {best}

Reuses the raw-space holdout predictions zarr if it already exists
(evaluate(compute_indices=False) produces it otherwise), so re-runs are
seconds, not minutes.
"""

import argparse
import logging
from pathlib import Path

import climx_project  # noqa: F401  (vendor + env + compat activation)

from climx_project import data, paths, preprocess, validation
from climx_project.metrics_weighting import coarsen_mask, decode_ar6_mask

logger = logging.getLogger(__name__)


def _annual_land_series(ds, var: str, land_mask):
    """Annual, land-mean series of `var` from a (time, lat, lon) dataset."""
    import numpy as np

    arr = ds[var].values.astype(np.float64)          # (T, H, W)
    years = ds.time.dt.year.values
    ys = np.unique(years)
    series = np.array([arr[years == y][:, land_mask].mean() for y in ys])
    return ys, series


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("unet", "worldmodel"))
    ap.add_argument("--checkpoint", default=None,
                    help="Lightning .ckpt path; default = the run's saved "
                         "results/models/<model>/<model>.ckpt")
    ap.add_argument("--predictions", default=None,
                    help="raw-space holdout predictions zarr; default = "
                         "results/predictions/<model>_holdout_predictions.zarr")
    ap.add_argument("--var", default="tas", choices=("tas", "tasmax", "tasmin", "pr"))
    ap.add_argument("--out", default=None,
                    help="PNG path; default = results/trend_diag_<model>.png")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    paths.ensure_dirs()
    pred_path = Path(args.predictions) if args.predictions else \
        paths.PREDICTIONS_DIR / f"{args.model}_holdout_predictions.zarr"

    if not pred_path.exists():
        # Produce the raw-space predictions via the starter's evaluate
        # (predict + reversal, no indices — the slow part skipped).
        from climx_project.holdout import _load_emulator
        dataset = data.open_datasets()
        pre = preprocess.load_preprocessed()
        X_train, y_train, metadata = pre["X_train"], pre["y_train"], pre["metadata"]
        emulator = _load_emulator(args.model, args.checkpoint,
                                  feature_vars=list(X_train.data_vars))
        X_hold, y_raw = validation.make_holdout_split(X_train, y_train, dataset)
        emulator.evaluate(
            X_hold, metadata,
            results_path=paths.INDICES_DIR / f"{args.model}_trend_evaluation.json",
            predictions_path=pred_path,
            historical_data_path=dataset.hist,
            TARGET_VARIABLES=paths.TARGET_VARIABLES,
            LOAD_PREDICTIONS=False, compute_indices=False)
    else:
        dataset = data.open_datasets()
        pre = preprocess.load_preprocessed()
        _, y_raw = validation.make_holdout_split(pre["X_train"], pre["y_train"],
                                                 dataset)

    pred = xr.open_zarr(pred_path)[[args.var]]
    gt = y_raw[[args.var]]
    # Land mask on the data's own grid (the scorer only sees land cells).
    mask = decode_ar6_mask()
    H, W = pred.sizes["lat"], pred.sizes["lon"]
    land = coarsen_mask(mask, H, W) >= 0

    yp, pred_series = _annual_land_series(pred, args.var, land)
    yg, gt_series = _annual_land_series(gt, args.var, land)
    assert (yp == yg).all(), "pred/gt year axes differ"

    slope_pred = np.polyfit(yp, pred_series, 1)[0] * 10   # K/decade
    slope_gt = np.polyfit(yg, gt_series, 1)[0] * 10
    bias = float((pred_series - gt_series).mean())
    corr = float(np.corrcoef(pred_series, gt_series)[0, 1])

    out_png = Path(args.out) if args.out else \
        paths.RESULTS_DIR / f"trend_diag_{args.model}_{args.var}.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(yg, gt_series, "o-", label=f"GT (slope {slope_gt:+.3f} K/dec)", lw=2)
    ax.plot(yp, pred_series, "s-", label=f"predicted (slope {slope_pred:+.3f} K/dec)", lw=2)
    ax.set_xlabel("year"); ax.set_ylabel(f"annual land-mean {args.var} (K)")
    ax.set_title(f"trend check — {args.model} on SSP370 holdout (corr {corr:.3f}, "
                 f"bias {bias:+.2f} K)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=110)
    logger.info("plot saved: %s", out_png)

    logger.info("GT slope:       %+.3f K/decade", slope_gt)
    logger.info("predicted slope:%+.3f K/decade  (ratio %.2f)",
                slope_pred, slope_pred / slope_gt if slope_gt else float("nan"))
    logger.info("mean bias: %+.2f K | annual correlation: %.3f", bias, corr)
    print(f"TREND CHECK ({args.model}, {args.var}): GT {slope_gt:+.3f} K/dec | "
          f"pred {slope_pred:+.3f} K/dec | ratio {slope_pred/slope_gt:.2f} | "
          f"bias {bias:+.2f} K | corr {corr:.3f} | plot {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
