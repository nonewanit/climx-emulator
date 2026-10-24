"""All directory constants, derived from this file's location.

Every path is __file__-relative on purpose: Hydra changes the process cwd to
its run dir, so any cwd-relative path would silently break. Nothing in this
repo may build paths from os.getcwd().
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LITE_DIR = DATA_DIR / "lite"
PREPROCESSING_DIR = DATA_DIR / "preprocessing_data_lite"

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
INDICES_DIR = RESULTS_DIR / "indices"
MODELS_DIR = RESULTS_DIR / "models"
SUBMISSIONS_DIR = RESULTS_DIR / "submissions"
SCORES_MD = RESULTS_DIR / "scores.md"

# Same order as the starter kit's preprocessing.TARGET_VARIABLES.
TARGET_VARIABLES = ["huss", "pr", "psl", "sfcWind", "tas", "tasmax", "tasmin"]

DATA_VERSION = "lite"
CLIMATOLOGY_NC = PREPROCESSING_DIR / "climatology_lite.nc"
SCALING_PARAMS_JSON = PREPROCESSING_DIR / "scaling_parameters_lite.json"
X_TRAIN_NC = PREPROCESSING_DIR / "X_train.nc"
Y_TRAIN_NC = PREPROCESSING_DIR / "y_train.nc"
X_TEST_NC = PREPROCESSING_DIR / "X_test.nc"


def ensure_dirs() -> None:
    """Create the artifact directories that must exist before a run."""
    for d in (RAW_DIR, LITE_DIR, PREPROCESSING_DIR, RUNS_DIR, PREDICTIONS_DIR,
              INDICES_DIR, MODELS_DIR, SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
