"""Lite-dataset acquisition and opening.

Two download routes:
- Kaggle CLI (primary): `kaggle competitions download -c climx-main-track`.
  Needs the user's API key at ~/.kaggle/kaggle.json. The bundle's internal
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
    """Download the competition data bundle into data/. Returns the zip path."""
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if BUNDLE_ZIP.exists() and BUNDLE_ZIP.stat().st_size > 0:
        logger.info("Kaggle bundle already present: %s", BUNDLE_ZIP)
        return BUNDLE_ZIP
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise RuntimeError(
            "No ~/.kaggle/kaggle.json found. Create a Kaggle API key "
            "(Account -> Settings -> Create New API Token), save it to "
            "~/.kaggle/kaggle.json (chmod 600), and accept the competition "
            "rules at https://www.kaggle.com/competitions/climx-main-track/rules. "
            "Fallback: `python -m climx_project.data download-hf`."
        )
    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(paths.DATA_DIR)]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return BUNDLE_ZIP


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
    if not paths.RAW_DIR.exists():
        return []
    return sorted(p for p in paths.RAW_DIR.rglob("*.nc"))


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
        raise FileNotFoundError(f"No .nc files found under {paths.RAW_DIR}")

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
    Note for smoke runs: `hist` may be replaced by a train slice by the caller
    (see scripts/fullres_smoke.py) to avoid reading the ~63 GB historical files
    just to exercise the code path.
    """
    import fsspec  # noqa: F401 (fsspec registers the hf:// protocol deps)
    import xarray as xr
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def open_one(suffix: str, n_days: int | None):
        url = f"{FULL_REPO}/{FULL_PREFIX}{suffix}.nc"
        f = fs.open(url, "rb")
        ds = xr.open_dataset(f, engine="h5netcdf", chunks=chunks or {})
        if n_days:
            ds = ds.isel(time=slice(0, n_days))
        return ds

    hist = xr.combine_by_coords(
        [open_one("hist_train_1850_1931", hist_days),
         open_one("hist_train_1932_2014", hist_days)],
        combine_attrs="drop_conflicts")
    train = xr.combine_by_coords(
        [open_one("train_ssp126", train_days), open_one("train_ssp370", train_days)],
        combine_attrs="drop_conflicts")
    test = open_one("test_forcings_ssp245", test_days)

    return type("FullDatasets", (), {"hist": hist, "train": train, "test_forcings": test})()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Lite dataset management")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download the Kaggle competition data bundle")
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
