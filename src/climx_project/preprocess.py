"""Wrappers over the starter kit's preprocess_train / preprocess_test.

Memory-safe by default: load_into_memory=False keeps everything dask-lazy and
the .nc files are written chunk-by-chunk (this machine has 7.5 GB RAM).

Writes into data/preprocessing_data_lite/:
    climatology_lite.nc            (by preprocess_train)
    scaling_parameters_lite.json   (by preprocess_train — CORRECT name; the
                                    starter's data_processing.py looks for the
                                    buggy 'scaling_params_lite.json', never run it)
    X_train.nc / y_train.nc / X_test.nc

CLI:
    python -m climx_project.preprocess        # run train + test preprocessing
    python -m climx_project.preprocess --train-only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from . import paths

logger = logging.getLogger(__name__)

_PREPROC_CHUNKS = {"time": 30, "lat": -1, "lon": -1}


def _save_chunked(ds: xr.Dataset, path: Path, *, encoding: dict | None = None) -> None:
    ds = ds.chunk(_PREPROC_CHUNKS)
    for var in ds.data_vars:
        ds[var].encoding.clear()
    logger.info("Writing %s ...", path.name)
    with ProgressBar():
        # compute=False returns a dask Delayed that must be explicitly
        # computed — discarding it writes only the file header, no data.
        ds.to_netcdf(path, encoding=encoding, compute=False).compute()
    logger.info("Wrote %s", path.name)


def run_preprocess_train(dataset, *, load_into_memory: bool = False,
                         stationarization_mode: str = "monthly",
                         version: str = paths.DATA_VERSION):
    from src.data_preprocessing.preprocessing import preprocess_train

    paths.PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X_train, y_train, metadata = preprocess_train(
        dataset,
        preprocessing_path=paths.PREPROCESSING_DIR,
        version=version,
        scaling_params_path=None,  # None -> compute + write scaling_parameters_lite.json
        load_into_memory=load_into_memory,
        stationarization_mode=stationarization_mode,
    )
    _save_chunked(X_train, paths.X_TRAIN_NC)
    _save_chunked(y_train, paths.Y_TRAIN_NC)
    logger.info("preprocess_train done in %.1f min", (time.time() - t0) / 60)
    return X_train, y_train, metadata


def run_preprocess_test(dataset, *, load_into_memory: bool = False,
                        stationarization_mode: str = "monthly",
                        version: str = paths.DATA_VERSION):
    from src.data_preprocessing.preprocessing import preprocess_test

    t0 = time.time()
    X_test, metadata = preprocess_test(
        dataset,
        preprocessing_path=paths.PREPROCESSING_DIR,
        version=version,
        scaling_params_path=paths.SCALING_PARAMS_JSON,
        load_into_memory=load_into_memory,
        stationarization_mode=stationarization_mode,
    )
    encoding = {var: {"zlib": False, "_FillValue": None} for var in X_test.data_vars}
    _save_chunked(X_test, paths.X_TEST_NC, encoding=encoding)
    logger.info("preprocess_test done in %.1f min", (time.time() - t0) / 60)
    return X_test, metadata


def load_metadata() -> dict:
    """Reconstruct the metadata dict the emulators expect, from the saved JSON."""
    with open(paths.SCALING_PARAMS_JSON) as f:
        scaling_params = json.load(f)
    return {
        "climatology_path": str(paths.CLIMATOLOGY_NC),
        "scaling_params_path": str(paths.SCALING_PARAMS_JSON),
        "scaling_params": scaling_params,
        "stationarization_mode": "monthly",
    }


def load_preprocessed() -> dict:
    """Load the saved X/y arrays (dask-backed). Raises with a hint if absent."""
    missing = [p.name for p in (paths.X_TRAIN_NC, paths.Y_TRAIN_NC, paths.X_TEST_NC)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Preprocessed data missing ({', '.join(missing)}). "
            "Run `python -m climx_project.preprocess` first."
        )
    return {
        "X_train": xr.open_dataset(paths.X_TRAIN_NC, chunks=_PREPROC_CHUNKS),
        "y_train": xr.open_dataset(paths.Y_TRAIN_NC, chunks=_PREPROC_CHUNKS),
        "X_test": xr.open_dataset(paths.X_TEST_NC, chunks=_PREPROC_CHUNKS),
        "metadata": load_metadata(),
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run ClimX lite preprocessing")
    parser.add_argument("--train-only", action="store_true", help="skip preprocess_test")
    args = parser.parse_args(argv)

    from .data import open_datasets

    dataset = open_datasets()
    run_preprocess_train(dataset)
    if not args.train_only:
        run_preprocess_test(dataset)


if __name__ == "__main__":
    main(sys.argv[1:])
