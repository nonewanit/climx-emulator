"""climx_project — shared plumbing for the ClimX course project.

Importing this package does two things, in order, and this is the ONLY place
either happens:

1. Activates the vendored starter kit (third_party/climx) by putting its root
   on sys.path. The starter mixes relative imports with repo-root-absolute
   imports (`from src.data_preprocessing...`), so its root must be importable
   as a top-level `src` package. index 0 beats cwd, so `import src` always
   resolves here regardless of Hydra's chdir.

2. Disables HDF5 file locking BEFORE any netCDF4 import (the starter does the
   same in train.py) to avoid deadlocks on shared filesystems.

Every script in this repo must start with `import climx_project`.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# The dask threads scheduler deadlocks in dask.array.store / xarray.to_netcdf
# on this machine (observed: every thread parked in futex_do_wait during the
# climatology write). The lite data is small (12x18 grids); the single-threaded
# scheduler is deterministic, uses less memory, and costs nothing in practice.
import dask  # noqa: E402

dask.config.set(scheduler="single-threaded")

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "climx"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from . import compat  # noqa: E402  (xarray strftime/ArrowStringArray patch)
from . import paths  # noqa: E402,F401  (imported after vendor activation)

compat.apply()

__all__ = ["paths"]
