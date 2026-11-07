"""Ensemble submission via unweighted index-map averaging (Phase D).

Competition rule: the ONLY allowed aggregation of indices across realizations
of the same emulator is an unweighted mean (main track). Since all our
parquets share the same deterministic row order (time-major, sorted lat/lon,
row_id 0..N-1), averaging their 15 index columns is exactly that operation.

Usage:
    python scripts/ensemble_indices.py -m unet-seed42 parquet_1.parquet [parquet_2 ...]
        -o results/submissions/submission_ensemble.parquet

All inputs must be realizations of the SAME model config (e.g. seeds of the
same experiment) — don't mix different architectures or epoch budgets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import climx_project  # noqa: F401  (vendor + env setup, as everywhere)
from climx_project.submission import verify_parquet

INDEX_COLUMNS = ["TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
                 "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"]


def average_parquets(paths: list[Path], out: Path) -> Path:
    if len(paths) < 2:
        raise ValueError("Need at least 2 parquets to average.")
    frames = [pd.read_parquet(p, engine="pyarrow") for p in paths]

    n = len(frames[0])
    cols = list(frames[0].columns)
    for f in frames[1:]:
        if len(f) != n or list(f.columns) != cols:
            raise ValueError(
                "Parquets disagree in shape/columns — they are not realizations "
                "of the same submission (or were written by different code)."
            )
    for f in frames:
        if not (f["row_id"].values == np.arange(n, dtype=np.int64)).all():
            raise ValueError(f"{f}: row_id is not 0..N-1")

    mean_idx = sum(f[INDEX_COLUMNS].astype(np.float64) for f in frames) / len(frames)
    out_df = pd.concat([frames[0][["row_id"]], mean_idx], axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    verify_parquet(out)
    print(f"Ensemble of {len(paths)} parquets -> {out} ({n} rows)")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Average index parquets into an ensemble submission")
    parser.add_argument("parquets", nargs="+", help="2+ submission parquets of the same model config")
    parser.add_argument("-o", "--out", required=True, help="output parquet path")
    args = parser.parse_args(argv)
    average_parquets([Path(p) for p in args.parquets], Path(args.out))


if __name__ == "__main__":
    main(sys.argv[1:])
