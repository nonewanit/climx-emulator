"""Train/eval preprocessing consistency check — %run in the Kaggle notebook.

Builds the model inputs for two sample years (2015, 2060) from the raw
ssp245 test forcings through BOTH pipelines and compares channel by channel:

  A. the starter's — daily_forcings + log_transform + aerosol split +
     scale_variables + seasonal features (exactly what the cachebuilder ran
     to build the training X stores), and
  B. the eval-side chunk preprocessor (fullres_eval.preprocess_chunk — what
     the 86-year predict feeds the model).

A channel differing beyond fp noise (|diff| > 1e-2) means the model sees
different inputs at predict time than it trained on — that would explain a
bad LB score no matter how many epochs you train.

Run:  %run scripts/check_preprocess.py [--years 2015,2060]
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

import climx_project  # noqa: F401
from climx_project import paths
from climx_project.fullres_eval import AEROSOLS, GHG, preprocess_chunk
from climx_project.models.unet import UNetEmulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("check_preprocess")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

RAW = Path("/tmp/raw_full/NorESM2-MM_r1i1p1f1_with_forcing_test_forcings_ssp245.nc")
SCALING_JSON = paths.PREPROCESSING_DIR / "scaling_parameters_full.json"
BUNDLED = paths.PROJECT_ROOT / "assets" / "fullres_scaling_parameters.json"
MODEL = Path("/kaggle/input/datasets/nonewanitchollakit/full-unet-weight/full_unet.pt")
CHUNKS = {"time": 512, "time_month": 24, "time_year": -1, "lat": -1, "lon": -1}


def ensure_raw() -> None:
    if RAW.exists():
        return
    from huggingface_hub import hf_hub_download
    import shutil
    log.info("downloading raw test forcings (1.4 GB) ...")
    p = hf_hub_download(
        repo_id="isp-uv-es/ClimX", repo_type="dataset",
        filename="full/NorESM2-MM_r1i1p1f1_with_forcing_test_forcings_ssp245.nc",
        local_dir=str(Path("/tmp/hf_tmp")))
    RAW.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(p, RAW)


def starter_side(ds: xr.Dataset, year: int, sc: dict) -> dict:
    """The starter's forcing pipeline on ONE year of raw data."""
    from climx_project.data import daily_forcings
    from src.data_preprocessing.preprocessing_utils import log_transform, scale_variables

    sub = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    x = daily_forcings(sub)                       # daily-up-sampled 10 forcings
    x = log_transform(x, [v for v in AEROSOLS if v in x], epsilon=1e-6)
    for v in AEROSOLS:
        if v not in x:
            continue
        x[f"{v}_positive"] = x[v].where(x[v] > 0, 0)
        x[f"{v}_negative"] = -(x[v].where(x[v] < 0, 0))
        x[f"{v}_zeros_mask"] = (x[v] == 0).astype(np.float32)
        x = x.drop_vars(v)
    x = scale_variables(x, sc)
    phase = 2 * np.pi * x.time.dt.dayofyear / 365.25
    x["sin_doy"] = np.sin(phase).expand_dims(lat=x.lat, lon=x.lon)
    x["cos_doy"] = np.cos(phase).expand_dims(lat=x.lat, lon=x.lon)
    return {k: np.asarray(x[k].values, dtype=np.float32) for k in x.data_vars}


def eval_side(ds: xr.Dataset, year: int, feature_order, sc) -> np.ndarray:
    """The eval-side chunk preprocessor on ONE year of raw data."""
    def to_float(ts):
        d = ts - ds["time"].values[0]
        return np.asarray([x.days + x.seconds / 86400 for x in d])

    time = ds["time"].values
    day_float = to_float(time)
    m_float = to_float(ds["time_month"].values)
    y_float = to_float(ds["time_year"].values)
    monthly = {v: ds[v].isel(forcing_scenario=0).values for v in AEROSOLS if v in ds}
    yearly = {v: ds[v].isel(forcing_scenario=0).values for v in GHG if v in ds}
    # noleap calendar: every year has 365 days, so plain index arithmetic
    t0 = (year - 2015) * 365
    t1 = min(t0 + 365, len(time))
    return preprocess_chunk(monthly, yearly, m_float, y_float, day_float,
                            time, t0, t1, feature_order, sc), t0, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2015,2060")
    args = ap.parse_args()

    ensure_raw()
    if not SCALING_JSON.exists() and BUNDLED.exists():
        SCALING_JSON.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(BUNDLED, SCALING_JSON)
    sc = json.load(open(SCALING_JSON))["scaling_params"]

    emulator = UNetEmulator(dict(
        n_in=22, n_out=7, base_channels=64, learning_rate=1e-3,
        variable_weights=[1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
        pad_to=[192, 288], skip_connections=True))
    emulator.load(MODEL)
    feature_order = list(emulator.feature_vars)
    log.info("feature order (%d): %s", len(feature_order), feature_order)

    ds = xr.open_dataset(RAW, engine="h5netcdf", chunks=CHUNKS)
    run = None
    try:
        from climx_project import monitoring
        run = monitoring.maybe_init_run(
            "check-preprocess", {"years": args.years})
    except Exception as e:  # wandb must never block the check
        log.warning("wandb init failed: %r", e)

    all_ok = True
    for year in [int(y) for y in args.years.split(",")]:
        log.info("=== year %d ===", year)
        starter = starter_side(ds, year, sc)
        X_eval, t0, t1 = eval_side(ds, year, feature_order, sc)
        ok_year = True
        for idx, name in enumerate(feature_order):
            a = starter.get(name)
            b = X_eval[:, idx]
            if a is None or a.shape != b.shape:
                log.error("%-22s MISSING/shape mismatch: %s vs %s",
                          name, None if a is None else a.shape, b.shape)
                ok_year = all_ok = False
                continue
            diff = np.abs(a.astype(np.float32) - b)
            md = float(diff.max())
            flag = "OK " if md < 1e-2 else "BAD"
            if md >= 1e-2:
                ok_year = all_ok = False
            log.info("%-22s %s max|diff| = %.3e  (mean %.3e)",
                     name, flag, md, float(diff.mean()))
            if run is not None:
                run.log({f"diff/{name}": md})
        log.info("year %d: %s", year, "CONSISTENT" if ok_year else "MISMATCH")
    log.info("VERDICT: %s", "PASS — eval inputs match training inputs" if all_ok
             else "FAIL — a channel differs; the model sees shifted inputs")
    if run is not None:
        run.log({"verdict_pass": int(all_ok)})
        run.finish()


if __name__ == "__main__":
    main()
