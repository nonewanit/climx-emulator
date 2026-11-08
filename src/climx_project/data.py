"""Lite-dataset acquisition and opening.

Two download routes:
- Kaggle CLI (primary): `kaggle competitions download -c climx-main-track`.
  Needs Kaggle credentials at ~/.kaggle/access_token (CLI v2.x) or the
  legacy ~/.kaggle/kaggle.json. The bundle's internal
  layout is NOT assumed — map_lite_files() inspects it and hardlinks whatever
  it finds into data/lite/ under the exact names the starter kit globs for.
- Hugging Face mirror (fallback): starter's get_dataset_from_hf with
  HF_ENDPOINT=https://hf-mirror.com (huggingface.co is blocked on this network).

CLI:
    python -m climx_project.data download   # download the Kaggle bundle
    python -m climx_project.data map-lite   # raw/ -> lite/ hardlinks + report
    python -m climx_project.data check      # verify the lite layout
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import xarray as xr

from . import paths

logger = logging.getLogger(__name__)

KAGGLE_COMPETITION = "climx-main-track"
BUNDLE_ZIP = paths.DATA_DIR / f"{KAGGLE_COMPETITION}.zip"

# logical name -> file-name fragment used to identify it in the raw bundle.
# Canonical output names match the starter's open_climx_virtual_datasets globs:
#   hist: "*hist_train_*.nc" | train: "*train_*.nc" (excluding hist) |
#   test: exactly one "*test_forcings_ssp245.nc"
LITE_FILES = {
    "hist_train_1850_1931": "hist",
    "hist_train_1932_2014": "hist",
    "train_ssp126": "ssp126",
    "train_ssp370": "ssp370",
    "train_ssp585": "ssp585",
    "test_forcings_ssp245": "ssp245",
}
LITE_FILE_PREFIX = "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_"


def _canonical_name(logical: str) -> str:
    return f"{LITE_FILE_PREFIX}{logical}.nc"


def download_kaggle_bundle(competition: str = KAGGLE_COMPETITION) -> Path:
    """Download the competition data bundle and extract it into data/raw/.

    Returns the zip path. Idempotent: skips the download when the zip exists
    and skips extraction when raw/ already holds .nc files.
    """
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (BUNDLE_ZIP.exists() and BUNDLE_ZIP.stat().st_size > 0):
        creds = Path.home() / ".kaggle"
        if not ((creds / "access_token").exists() or (creds / "kaggle.json").exists()):
            raise RuntimeError(
                "No Kaggle credentials found in ~/.kaggle/. Create an API token "
                "(Account -> Settings -> Create New API Token) and save it to "
                "~/.kaggle/access_token (CLI v2.x; legacy name kaggle.json), "
                "chmod 600, and accept the competition rules at "
                "https://www.kaggle.com/competitions/climx-main-track/rules. "
                "Fallback: `python -m climx_project.data download-hf`."
            )
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(paths.DATA_DIR)]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    _extract_bundle()
    return BUNDLE_ZIP


def _extract_bundle() -> None:
    """Unzip the downloaded bundle into data/raw/ (idempotent).

    The zip's internal layout is NOT assumed — map_lite_files() rglobs
    data/raw recursively, so any nesting works.
    """
    import zipfile

    paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if list(paths.RAW_DIR.rglob("*.nc")):
        logger.info("Bundle already extracted (raw/ has .nc files); skipping")
        return
    logger.info("Extracting %s -> %s ...", BUNDLE_ZIP.name, paths.RAW_DIR)
    with zipfile.ZipFile(BUNDLE_ZIP) as zf:
        zf.extractall(paths.RAW_DIR)
    n = len(list(paths.RAW_DIR.rglob("*.nc")))
    logger.info("Extracted %d .nc files into %s", n, paths.RAW_DIR)


HF_MIRROR = "https://hf-mirror.com"  # huggingface.co is blocked on this network
HF_REPO = "isp-uv-es/ClimX"

# Everything the starter's open_climx_virtual_datasets globs for, plus the
# convenience forcings_all.nc (26 MB — kept for parity with the HF layout).
HF_LITE_FILES = (
    "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_hist_train_1850_1931.nc",
    "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_hist_train_1932_2014.nc",
    "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_train_ssp126.nc",
    "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_train_ssp370.nc",
    "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_train_ssp585.nc",
    "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_test_forcings_ssp245.nc",
    "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_forcings_all.nc",
)


def download_hf_lite() -> None:
    """Fallback route: plain curl -4 against the HF mirror.

    huggingface_hub's HTTP client prefers IPv6, which this machine cannot
    route — curl -4 forces IPv4. Resumable (-C -) and idempotent.
    """
    paths.LITE_DIR.mkdir(parents=True, exist_ok=True)
    for name in HF_LITE_FILES:
        target = paths.LITE_DIR / name
        if target.exists() and target.stat().st_size > 0:
            logger.info("Already present: %s", name)
            continue
        url = f"{HF_MIRROR}/datasets/{HF_REPO}/resolve/main/lite/{name}"
        cmd = ["curl", "-4", "-L", "-C", "-", "--retry", "3",
               "--connect-timeout", "20", "-o", str(target), url]
        logger.info("Downloading %s ...", name)
        result = subprocess.run(cmd)
        if result.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            raise RuntimeError(f"Download failed for {name} (curl exit {result.returncode})")
    logger.info("Lite download complete.")


def _nc_candidates() -> list[Path]:
    """All .nc files under data/ except the preprocessing outputs.

    Scans the whole data/ tree (not just raw/): covers teammates who
    extracted the bundle somewhere else (e.g. straight into data/ or
    data/lite/) and still lets map-lite pick the files up.
    """
    if not paths.DATA_DIR.exists():
        return []
    excluded = paths.PREPROCESSING_DIR.resolve()
    return sorted(p for p in paths.DATA_DIR.rglob("*.nc")
                  if not p.is_relative_to(excluded))


def _grid_is_lite(path: Path) -> bool:
    """True if the file's lat/lon grid is the 16x-coarsened 12x18 lite grid."""
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            return ds.sizes.get("lat", 0) == 12 and ds.sizes.get("lon", 0) == 18
    except Exception as e:  # pragma: no cover - best effort inspection
        logger.warning("Could not inspect %s: %s", path, e)
        return False


def map_lite_files(force: bool = False) -> dict[str, Path]:
    """Hardlink raw-bundle NetCDFs into data/lite/ under canonical starter names.

    Returns {logical_name: source_path}. Never copies: hardlinks cost no disk.
    """
    if not paths.RAW_DIR.exists() and not list(paths.LITE_DIR.glob("*.nc")):
        raise FileNotFoundError(
            f"Nothing to map: {paths.RAW_DIR} missing and {paths.LITE_DIR} empty. "
            "Run `python -m climx_project.data download` first."
        )
    paths.LITE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = _nc_candidates()
    if not candidates and not list(paths.LITE_DIR.glob("*.nc")):
        raise FileNotFoundError(
            f"No .nc files found under {paths.DATA_DIR} (excluding "
            f"{paths.PREPROCESSING_DIR}). Run `python -m climx_project.data "
            "download` to fetch and extract the Kaggle bundle, or put the "
            "unzipped competition data anywhere under data/ and retry.")

    # Norm for matching: lowercase, '-' -> '_'
    def norm(name: str) -> str:
        return name.lower().replace("-", "_")

    report: dict[str, Path] = {}
    for logical, fragment in LITE_FILES.items():
        target = paths.LITE_DIR / _canonical_name(logical)
        if target.exists() and not force:
            logger.info("Already mapped: %s", target)
            report[logical] = target
            continue

        # 1. Exact canonical name in raw/ (bundle may already use starter names)
        exact = paths.RAW_DIR / _canonical_name(logical)
        if exact.exists():
            src = exact
        else:
            # 2. Fragment match (unique per logical name for the *_ssp* files)
            matches = [p for p in candidates if fragment in norm(p.name)]
            if len(matches) == 1:
                src = matches[0]
            else:
                # The two hist files share the 'hist' fragment: disambiguate by year.
                year = {"hist_train_1850_1931": "1850", "hist_train_1932_2014": "1932"}.get(logical)
                hist_matches = [p for p in matches if year in p.name] if year else []
                if len(hist_matches) == 1:
                    src = hist_matches[0]
                else:
                    logger.error("Cannot identify %s: candidates=%s", logical, [p.name for p in matches])
                    continue

        if not _grid_is_lite(src):
            logger.warning("%s does not look like a 12x18 lite grid; skipping", src.name)
            continue
        try:
            os.link(src, target)
        except FileExistsError:
            target.unlink()
            os.link(src, target)
        except OSError as e:  # e.g. cross-device or FS without hardlinks
            import shutil
            logger.info("hardlink failed (%s) — copying instead", e)
            shutil.copy2(src, target)
        report[logical] = src
        logger.info("Mapped %s -> %s", src.name, target.name)

    return report


def check_lite_files() -> dict[str, int]:
    """Verify data/lite/ has all six files with lite grids. Returns {name: bytes}."""
    problems = []
    sizes: dict[str, int] = {}
    for logical in LITE_FILES:
        path = paths.LITE_DIR / _canonical_name(logical)
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        sizes[path.name] = path.stat().st_size
        if not _grid_is_lite(path):
            problems.append(f"{path.name}: not a 12x18 grid")
    if problems:
        raise RuntimeError("Lite data check failed:\n  " + "\n  ".join(problems))
    return sizes


def open_datasets(chunks: object = "auto") -> object:
    """Open the lite datasets the way the starter kit expects.

    Returns the starter's ClimXVirtualDatasets dataclass with attributes
    .hist / .train / .test_forcings (lazily loaded xarray datasets).
    Honors the CLIMX_LITE_DIR override (e.g. a Kaggle input mount).
    """
    from src.utils.hugging_face_utils import open_climx_virtual_datasets

    check_lite_files()
    return open_climx_virtual_datasets(
        paths.LITE_DIR.parent, paths.LITE_DIR.name, parallel=True, chunks=chunks)


FULL_REPO = "datasets/isp-uv-es/ClimX/full"
FULL_PREFIX = "NorESM2-MM_r1i1p1f1_with_forcing_"


def open_full_datasets(*, hist_days: int | None = None, train_days: int | None = None,
                       test_days: int | None = None, chunks: dict | None = None) -> object:
    """Open the full-resolution (192x288) NetCDFs REMOTELY from Hugging Face.

    Streams HDF5 chunks over range requests via the HF filesystem — nothing is
    downloaded. The optional *_days slices bound the time axis for smoke runs
    (each slice reads only the chunks it touches).

    Returns a namespace with .hist / .train / .test_forcings (lazily loaded).
    Real historical files for `hist` (smoke scripts may override it with a
    train slice); `*_days` slice the time axis for smoke runs.

    Requires the SINGLE-THREADED dask scheduler (the package default) — the
    threads scheduler races on the shared remote file handle (v5 symptom:
    hours of tiny re-reads with no progress).
    """
    import fsspec  # noqa: F401 (fsspec registers the hf:// protocol deps)
    import xarray as xr
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def open_one(suffix: str, n_days: int | None):
        url = f"{FULL_REPO}/{FULL_PREFIX}{suffix}.nc"
        # Time-chunking is ESSENTIAL for remote streaming: with chunks={} the
        # whole (multi-GB) variable is one dask chunk and every access pulls
        # the entire file over HTTP. ~1 year per chunk = ~80 MB per var/read.
        # NOTE: the file object is shared across dask tasks — keep the dask
        # scheduler SINGLE-THREADED (the package default; the threads
        # scheduler races seeks on the shared HTTP handle, cf. v5's thrash).
        f = fs.open(url, "rb")
        ds = xr.open_dataset(f, engine="h5netcdf", chunks=chunks or {"time": 365})
        if n_days:
            ds = ds.isel(time=slice(0, n_days))
        return ds

    # The scenario files share an IDENTICAL daily time axis (2015-2100) — they
    # must be concatenated along a NEW forcing_scenario dimension (the
    # starter's convention), never combine_by_coords (which would try to
    # align the duplicate times). The hist files cover sequential periods, so
    # plain time-concat is correct. All opens are lazy (metadata only);
    # smoke scripts may override `hist` with a train slice to skip the
    # ~63 GB historical stream.
    hist = xr.concat(
        [open_one("hist_train_1850_1931", hist_days),
         open_one("hist_train_1932_2014", hist_days)],
        dim="time", combine_attrs="drop_conflicts")
    train = xr.concat(
        [open_one("train_ssp126", train_days), open_one("train_ssp370", train_days)],
        dim="forcing_scenario", combine_attrs="drop_conflicts")
    test = open_one("test_forcings_ssp245", test_days)

    return type("FullDatasets", (), {"hist": hist, "train": train, "test_forcings": test})()


def daily_forcings(ds: "xr.Dataset") -> "xr.Dataset":
    """Upsample native-resolution forcings onto the daily time grid.

    The FULL-resolution files store forcings only on their native
    time_year/time_month grids, but the starter's preprocessing (climatology,
    stationarization) expects a daily 'time' dim — the lite files had daily
    forcings pre-merged, which is why this never surfaced on lite. Applies the
    starter's own upsample_forcing_to_daily and drops the native aux dims.
    """
    import xarray as xr
    from src.data_preprocessing.preprocessing import FORCING_VARIABLES, TARGET_VARIABLES
    from src.data_preprocessing.preprocessing_utils import upsample_forcing_to_daily

    yearly = [v for v in FORCING_VARIABLES
              if v in ds and ("time_year" in ds[v].dims or "year" in ds[v].dims)]
    monthly = [v for v in FORCING_VARIABLES
               if v in ds and "time_month" in ds[v].dims]

    def clean_forcing(varlist):
        """Select forcing vars and strip redundant axes before upsampling.

        The historical files store native-res forcings with a CONSTANT daily
        'time' axis (verified: every time-slice identical — the value depends
        only on the native time_year/time_month index) and sometimes a
        degenerate forcing_scenario of size 1. The starter's
        upsample_forcing_to_daily cannot handle the extra 'time' dim (its
        final rename then collides with the real daily time), so drop it:
        isel(time=0) is lossless by the constancy above.
        """
        out = {}
        for v in varlist:
            da = ds[v]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            if "forcing_scenario" in da.dims and da.sizes["forcing_scenario"] == 1:
                da = da.isel(forcing_scenario=0, drop=True)
            out[v] = da
        cleaned = xr.Dataset(out)
        # upsample_forcing_to_daily interps onto ds.time.values — carry the
        # daily time axis along as a coordinate.
        if "time" in ds.coords:
            cleaned = cleaned.assign_coords(time=ds.time)
        return cleaned

    parts = []
    targets = [v for v in TARGET_VARIABLES if v in ds]
    if targets:
        parts.append(ds[targets])
    if yearly:
        parts.append(upsample_forcing_to_daily(clean_forcing(yearly), yearly, "time_year"))
    if monthly:
        parts.append(upsample_forcing_to_daily(clean_forcing(monthly), monthly, "time_month"))
    if not parts:
        return ds

    out = xr.merge(parts, combine_attrs="drop_conflicts")
    out = out.drop_vars(["time_month", "time_year"], errors="ignore")
    for dim in ("time_month", "time_year"):
        if dim in out.dims:
            out = out.isel({dim: 0}, drop=True)
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Lite dataset management")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download the Kaggle competition data bundle and extract it into data/raw/")
    sub.add_parser("download-hf", help="fallback: download lite/ from the HF mirror")
    sub.add_parser("map-lite", help="hardlink raw bundle files into data/lite/")
    sub.add_parser("check", help="verify the data/lite/ layout")
    args = parser.parse_args(argv)

    if args.cmd == "download":
        download_kaggle_bundle()
    elif args.cmd == "download-hf":
        download_hf_lite()
        map_lite_files()
    elif args.cmd == "map-lite":
        report = map_lite_files()
        for logical, src in report.items():
            print(f"{logical}: {src}")
    elif args.cmd == "check":
        sizes = check_lite_files()
        total = sum(sizes.values())
        for name, size in sizes.items():
            print(f"{name}: {size / 1e6:.1f} MB")
        print(f"total: {total / 1e6:.1f} MB — OK")


if __name__ == "__main__":
    main(sys.argv[1:])
