"""Full-res LOCAL holdout: ssp370 2081-2100, scored with the official rnNSE.

Uses ONLY attached cache data (X stores for inputs read from the mount,
reversed y stores as GT, hist from the extras mount) — no raw downloads.
Predicts with the saved model, reverses with the exact pass-1 formulas
(log cap -3, fp16 store), then runs the same index -> format -> upscale ->
rnNSE chain as the submission — but on a scenario the model TRAINED on.

Reading the score:
  healthy here + bad LB  -> scenario generalization gap (train on all SSPs)
  bad here too           -> undertraining / model problem (more epochs, LR)

Run:  %run scripts/fullres_holdout.py [--model <path>] [--years 2081,2100]
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

import climx_project  # noqa: F401
from climx_project import paths, validation
from climx_project.fullres_eval import compute_indices_eagerly, open_hist_from_mount
from climx_project.models.unet import UNetEmulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fullres_holdout")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

CACHE_DIR = Path(os.environ.get("CLIMX_CACHE_DIR", "/tmp/full_cache"))
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
BUNDLED = paths.PROJECT_ROOT / "assets" / "fullres_scaling_parameters.json"
MODEL = Path("/kaggle/input/datasets/nonewanitchollakit/full-unet-weight/full_unet.pt")
INDEX_VARS = ["tas", "tasmax", "tasmin", "pr"]   # the 15 indices only need these
TIME_CHUNK = 256


def find_mount_dirs():
    return [d for d in os.environ.get("CLIMX_CACHE_INPUT_DIRS", "").split(":") if d]


def open_x_store(mounts, channel: str):
    for d in mounts:
        p = Path(d) / f"X_{channel}_ssp370.zarr"
        if p.is_dir():
            return zarr.open_group(str(p), mode="r")
    raise FileNotFoundError(f"X_{channel}_ssp370.zarr not found in {mounts}")


def open_y_store(mounts, var: str):
    for d in mounts:
        p = Path(d) / f"y_{var}_ssp370.zarr"
        if p.is_dir():
            return zarr.open_group(str(p), mode="r")
    raise FileNotFoundError(f"y_{var}_ssp370.zarr not found in {mounts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--years", default="2081,2100")
    args = ap.parse_args()
    y0, y1 = [int(x) for x in args.years.split(",")]

    if not SCALING_JSON.exists() and BUNDLED.exists():
        import shutil
        SCALING_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED, SCALING_JSON)
    scaling = json.load(open(SCALING_JSON))
    sc = scaling["scaling_params"]
    log_info = scaling.get("_log_transform_info", {})

    mounts = find_mount_dirs()
    log.info("mounts: %s", mounts)

    # ---------- model ----------
    emulator = UNetEmulator(dict(
        n_in=22, n_out=7, base_channels=64, learning_rate=1e-3,
        variable_weights=[1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
        pad_to=[192, 288], skip_connections=True))
    emulator.load(args.model)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator.model = emulator.model.to(device)
    emulator.model.eval()
    feature_order = list(emulator.feature_vars)
    target_vars = list(emulator.target_vars)
    log.info("model on %s; feature_vars=%s; target_vars=%s", device,
             feature_order, target_vars)

    # ---------- X from the cache stores (read from the mount) ----------
    xg = {ch: open_x_store(mounts, ch) for ch in feature_order}
    time_c = xg[feature_order[0]]["time"][:]
    # noleap calendar (365 days/year, stores start 2015-01-01): plain math
    i0 = (y0 - 2015) * 365
    i1 = min((y1 + 1 - 2015) * 365, len(time_c))
    n_days = i1 - i0
    nlat, nlon = xg[feature_order[0]][feature_order[0]].shape[1:]
    log.info("holdout %d-%d: %d days", y0, y1, n_days)

    # ---------- predict + reverse (pass-1 formulas) ----------
    preds = np.empty((n_days, len(INDEX_VARS), nlat, nlon), dtype=np.float16)
    t_all = time.time()
    for c, t0 in enumerate(range(0, n_days, TIME_CHUNK)):
        t1 = min(n_days, t0 + TIME_CHUNK)
        X = np.empty((t1 - t0, len(feature_order), nlat, nlon), dtype=np.float32)
        for ci, ch in enumerate(feature_order):
            X[:, ci] = xg[ch][ch][i0 + t0:i0 + t1].astype(np.float32)
        outs = []
        for s0 in range(0, len(X), 32):
            sub = torch.from_numpy(X[s0:s0 + 32]).to(device)
            with torch.no_grad():
                outs.append(emulator.model(sub).cpu().numpy())
        pred = np.concatenate(outs, axis=0).astype(np.float32)
        del X, outs
        for vi, v in enumerate(target_vars):
            if v not in INDEX_VARS:
                continue
            vals = pred[:, vi] * sc[v]["std"] + sc[v]["mean"]
            if v in log_info:
                vals = np.minimum(vals, -3.0)
                vals = (10 ** vals) - log_info[v]["epsilon"]
            np.clip(vals, -65504.0, 65504.0, out=vals)
            preds[t0:t1, INDEX_VARS.index(v)] = vals.astype(np.float16)
        del pred, vals
        if c % 10 == 0:
            log.info("chunk %d/%d", c + 1, -(-n_days // TIME_CHUNK))
    log.info("predict+reverse: %.1f min", (time.time() - t_all) / 60)
    del xg

    # ---------- GT: reverse the y cache (exact inverse of write_cache_y) ----------
    gt_vars = {}
    lat = lon = None
    for var in INDEX_VARS:
        g = open_y_store(mounts, var)
        yv = g[var][i0:i1].astype(np.float32)
        if var in log_info:
            raw = 10 ** (yv * sc[var]["std"] + sc[var]["mean"]) - log_info[var]["epsilon"]
        else:
            raw = yv * sc[var]["std"] + sc[var]["mean"]
        gt_vars[var] = raw
        lat, lon = g["lat"][:], g["lon"][:]
        del yv, raw
    time_cut = time_c[i0:i1]   # cftime objects — fine for xarray/index code
    pred_ds = xr.Dataset(
        {v: (("time", "lat", "lon"), preds[:, INDEX_VARS.index(v)])
         for v in INDEX_VARS},
        coords={"time": time_cut, "lat": lat, "lon": lon})
    gt_ds = xr.Dataset(
        {v: (("time", "lat", "lon"), gt_vars[v]) for v in INDEX_VARS},
        coords={"time": time_cut, "lat": lat, "lon": lon})
    del preds, gt_vars

    # ---------- indices (the patched SELECTED_INDICES) ----------
    hist = open_hist_from_mount(mounts, CACHE_DIR)
    pred_idx = compute_indices_eagerly(
        validation.compute_gt_indices(pred_ds, hist))
    gt_idx = compute_indices_eagerly(
        validation.compute_gt_indices(gt_ds, hist))
    del hist, pred_ds, gt_ds

    # ---------- format -> upscale -> rnNSE (the submission chain) ----------
    ds_p = validation.upscale(validation._format_and_dataset(pred_idx, "holdout_unet"))
    ds_g = validation.upscale(validation._format_and_dataset(gt_idx, "holdout_gt"))
    pred_df = validation.to_df(ds_p)
    gt_df = validation.to_df(ds_g)
    score = validation.score_local(gt_df, pred_df)
    log.info("LOCAL HOLDOUT nNSE (ssp370 %d-%d): %.5f", y0, y1, score)
    for name in gt_df.columns:
        if name == "row_id":
            continue
        log.info("  %-7s pred [%.4g, %.4g]  gt [%.4g, %.4g]",
                 name, pred_df[name].min(), pred_df[name].max(),
                 gt_df[name].min(), gt_df[name].max())


if __name__ == "__main__":
    main()
