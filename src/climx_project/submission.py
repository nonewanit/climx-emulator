"""Kaggle submission construction, verification, and upload.

Path: indices dict -> format_indices -> drop forcing_scenario -> Dataset ->
upscale 12x18 -> 192x288 -> write_kaggle_submission_parquet -> read-back check.

The starter's parquet writer SILENTLY zero-fills NaNs (with a warning), which
would hide a real bug and produce a wrong submission — so we check for
NaN/Inf BEFORE writing and fail loudly, and double-check afterwards.

CLI:
    python -m climx_project.submission verify <file.parquet>
    python -m climx_project.submission submit <file.parquet> "<message>"
"""

import argparse
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import paths
from .validation import upscale

logger = logging.getLogger(__name__)

EXPECTED_ROWS = 86 * 192 * 288  # 4,755,456 — 86 test years x full grid
KAGGLE_COMPETITION = "climx-main-track"


def build_submission(indices_pred: dict[str, xr.DataArray], model: str) -> Path:
    """Format, upscale, and write the submission parquet. Returns its path."""
    import src.utils.kaggle_submission as ks
    from src.utils.indices_utils import format_indices

    formatted = format_indices(indices_pred, model, paths.DATA_VERSION, save_indices=False)
    for k, v in formatted.items():
        if "forcing_scenario" in v.dims:
            formatted[k] = v.isel(forcing_scenario=0, drop=True)

    ds = xr.Dataset(formatted)
    ds = upscale(ds)

    # Fail loudly BEFORE the writer can silently zero-fill NaNs.
    for k in ds.data_vars:
        if bool(ds[k].isnull().any().item()):
            raise ValueError(f"Index {k!r} contains NaN after upscaling — refusing to write.")
        if not np.isfinite(ds[k].values).all():
            raise ValueError(f"Index {k!r} contains non-finite values — refusing to write.")

    out_path = paths.SUBMISSIONS_DIR / f"submission_{model}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ks.write_kaggle_submission_parquet(ds, out_path)
    for w in caught:
        if "NaN" in str(w.message):
            raise RuntimeError(f"Parquet writer zero-filled NaNs: {w.message}")

    verify_parquet(out_path)
    logger.info("Submission written: %s", out_path)
    return out_path


def verify_parquet(path: Path, expected_rows: int = EXPECTED_ROWS) -> dict:
    """Read the parquet back and check every invariant Kaggle will enforce."""
    import src.utils.kaggle_submission as ks

    path = Path(path)
    df = pd.read_parquet(path, engine="pyarrow")
    expected_cols = ["row_id"] + list(ks.CLIMX_INDEX_COLUMNS)

    report: dict = {"file": path.name, "rows": len(df), "columns": list(df.columns)}
    problems: list[str] = []
    if list(df.columns) != expected_cols:
        problems.append(f"columns mismatch: {list(df.columns)} != {expected_cols}")
    if len(df) != expected_rows:
        problems.append(f"row count {len(df)} != expected {expected_rows}")
    if len(df) and not (df["row_id"].values == np.arange(len(df), dtype=np.int64)).all():
        problems.append("row_id is not 0..N-1")
    value_cols = [c for c in df.columns if c != "row_id"]
    if not np.isfinite(df[value_cols].to_numpy(dtype=np.float64)).all():
        problems.append("non-finite values present")

    if problems:
        raise RuntimeError(f"Verification of {path.name} failed:\n  " + "\n  ".join(problems))
    report["ok"] = True
    logger.info("Parquet verified: %d rows, %d columns, finite", report["rows"], len(report["columns"]))
    return report


def submit(path: Path, message: str, competition: str = KAGGLE_COMPETITION) -> None:
    """Upload via the Kaggle CLI. Falls back to a hint about the web UI on error."""
    cmd = ["kaggle", "competitions", "submit", "-c", competition,
           "-f", str(path), "-m", message]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle CLI exited with {result.returncode}. Check your API credentials "
            "and team registration; or upload the file manually on "
            f"https://www.kaggle.com/competitions/{competition}/submissions"
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="ClimX submission tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_verify = sub.add_parser("verify", help="verify a submission parquet")
    p_verify.add_argument("file")
    p_submit = sub.add_parser("submit", help="upload a submission parquet via the Kaggle CLI")
    p_submit.add_argument("file")
    p_submit.add_argument("message")
    args = parser.parse_args(argv)

    if args.cmd == "verify":
        report = verify_parquet(Path(args.file))
        print(report)
    elif args.cmd == "submit":
        submit(Path(args.file), args.message)


if __name__ == "__main__":
    main(sys.argv[1:])
