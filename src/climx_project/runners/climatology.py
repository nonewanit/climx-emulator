"""Climatology baseline runner: no training — the model IS the preprocessed
monthly climatology (preprocess_train must have run first). Also runs the
~1.0 self-check that validates the scoring plumbing end to end."""

import logging

from .. import paths, validation
from . import base

logger = logging.getLogger(__name__)


def run(cfg, dataset, X_train, y_train, metadata, X_test) -> dict:
    from src.models.climatology_model import ClimatologyBaseline

    emulator = ClimatologyBaseline(climatology_path=str(paths.CLIMATOLOGY_NC))
    emulator.fit(X_train, y_train, {})  # opens the climatology file

    # Wiring self-check before the real run: score the model against itself.
    X_hold, y_raw = validation.make_holdout_split(X_train, y_train, dataset)
    self_score = validation.self_check_climatology(emulator, X_hold, metadata, dataset.hist)
    if not (0.999 <= self_score <= 1.001):
        logger.warning("Self-check score %.6f is not ~1.0 — plumbing may be broken.", self_score)

    result = base.run_experiment(emulator, "climatology", dataset, X_train, y_train,
                                 metadata, X_test)
    result["self_check"] = self_score
    return result
