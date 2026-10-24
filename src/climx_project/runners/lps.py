"""Linear Pattern Scaling baseline runner: global CO2 -> global-mean tas least
squares, then per-pixel tas -> per-variable least squares. No Lightning
trainer; fits in minutes on CPU."""

import logging

from .. import paths
from . import base

logger = logging.getLogger(__name__)


def run(cfg, dataset, X_train, y_train, metadata, X_test) -> dict:
    from src.models.lps_model import LPSBaseline

    emulator = LPSBaseline()
    emulator.fit(
        X_train, y_train,
        co2_var=cfg.fit.co2_var,
        tas_var=cfg.fit.tas_var,
    )
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    emulator.save(paths.MODELS_DIR / "lps.pt")

    return base.run_experiment(emulator, "lps", dataset, X_train, y_train,
                               metadata, X_test)
