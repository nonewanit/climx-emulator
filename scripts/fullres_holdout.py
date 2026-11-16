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


def open_pkg_store(mounts, name: str, cache_dir: Path):
    """Open <name>.zarr.zarrpkg from the mount — directly through zarr's
    ZipStore (reads stream sequentially through the zip, no ~24 GB
    extraction). Packages whose chunked writers omitted the root group
    metadata fall back to extract+repair in cache_dir (the same repair
    copy_datasets_to_cache does)."""
    for d in mounts:
        p = Path(d) / f"{name}.zarr.zarrpkg"
        if not p.is_file():
            continue
        try:
            return zarr.open_zip(str(p), mode="r")
        except Exception:
            pass  # missing root metadata or old zarr — extract below
        dst = cache_dir / f"{name}.zarr"
        if not (dst / "zarr.json").exists() and not (dst / ".zgroup").exists():
            if dst.exists():
                import shutil
                shutil.rmtree(dst)
            import zipfile
            import json
            log.info("extracting %s (root metadata missing in the package)",
                     p.name)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(dst)
            if not (dst / "zarr.json").exists() and not (dst / ".zgroup").exists():
                (dst / "zarr.json").write_text(json.dumps(
                    {"zarr_format": 3, "node_type": "group", "attributes": {}}))
        return zarr.open_group(str(dst), mode="r")
    raise FileNotFoundError(f"{name}.zarr.zarrpkg not found in {mounts}")


def read_time_coord(g):
    """Decode the store's raw time array back to a CFTimeIndex (the
    cachebuilder's to_zarr wrote it with CF units/calendar attrs; reading
    it raw gives undecoded ints that break xclim's resample)."""
    from xarray.coding.times import decode_cf_datetime

    t = g["time"]
    vals = np.asarray(t[:])
    units = t.attrs.get("units")
    if units is None:
        raise ValueError(f"time array has no CF units attr (attrs={dict(t.attrs)})")
    calendar = t.attrs.get("calendar", "standard")
    idx = xr.CFTimeIndex(decode_cf_datetime(vals, units=units,
                                            calendar=calendar,
                                            use_cftime=True))
    return idx


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
    if str(args.model).endswith(".ckpt"):
        # Lightning periodic snapshot (epoch-NN.ckpt) — a different format
        # from emulator.save's full_unet.pt. Rebuild the channel order from
        # the cache package names, exactly like the cache fit does.
        from climx_project.models.unet import UNetModule, TARGET_VARIABLE_ORDER
        emulator.model = UNetModule.load_from_checkpoint(args.model)
        names = None
        for d in mounts:
            pkgs = sorted(Path(d).glob("X_*_ssp370.zarr.zarrpkg"))
            if pkgs:
                names = [p.name.split("_", 1)[1].rsplit("_", 1)[0]
                         for p in pkgs]
                break
        if not names:
            raise FileNotFoundError(
                "no X_*_ssp370.zarr.zarrpkg packages in the mounts — cannot "
                "derive the channel order for this checkpoint")
        emulator.feature_vars = names
        emulator.target_vars = list(TARGET_VARIABLE_ORDER)
    else:
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    xg = {ch: open_pkg_store(mounts, f"X_{ch}_ssp370", CACHE_DIR)
          for ch in feature_order}
    time_c = read_time_coord(xg[feature_order[0]])
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
        g = open_pkg_store(mounts, f"y_{var}_ssp370", CACHE_DIR)
        if g["time"].shape[0] < i1:
            raise ValueError(
                f"y_{var}_ssp370 has {g['time'].shape[0]} days — shorter "
                f"than the X span ({i1}); the stores are misaligned")
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

    # ---------- trend check: does the model capture the forced warming? ----------
    # The scorer's per-cell R2 = 1 - MSE/Var over the test period is won or
    # lost on the warming TREND in the temperature indices: slope ratio ~1
    # means the model extracts the GHG signal; ~0 means no epoch count will
    # fix the LB (the forcing encoding needs work instead).
    try:
        from climx_project.metrics_weighting import decode_ar6_mask
        land = decode_ar6_mask() >= 0
        years = np.array([t.year for t in time_cut.values])
        ys = np.unique(years)
        g = np.array([gt_ds["tas"].values[years == y][:, land].mean() for y in ys])
        p = np.array([pred_ds["tas"].values[years == y][:, land].mean() for y in ys])
        sg = np.polyfit(ys, g, 1)[0] * 10
        sp = np.polyfit(ys, p, 1)[0] * 10
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(ys, g, "o-", label=f"GT ({sg:+.3f} K/dec)")
        ax.plot(ys, p, "s-", label=f"predicted ({sp:+.3f} K/dec)")
        ax.set_xlabel("year")
        ax.set_ylabel("annual land-mean tas (K)")
        ax.set_title(f"trend check — fullres {Path(args.model).name}")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        png = paths.RESULTS_DIR / "trend_diag_fullres_tas.png"
        fig.savefig(png, dpi=110)
        log.info("TREND: GT %+.3f K/dec | pred %+.3f K/dec | ratio %.2f | "
                 "corr %.3f | plot %s",
                 sg, sp, sp / sg if sg else float("nan"),
                 float(np.corrcoef(p, g)[0, 1]), png)
    except Exception as e:  # non-fatal: the score below is what matters
        log.warning("trend check failed (non-fatal): %r", e)

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
    run = None
    try:
        from climx_project import monitoring
        run = monitoring.maybe_init_run(
            "holdout-ssp370", {"years": f"{y0}-{y1}",
                               "model": str(args.model)})
        if run is not None:
            run.log({"local_nNSE": float(score)})
            run.log({f"range/{name}": float(pred_df[name].mean())
                     for name in gt_df.columns if name != "row_id"})
            run.finish()
    except Exception as e:  # wandb must never block the eval
        log.warning("wandb log failed: %r", e)


if __name__ == "__main__":
    main()
