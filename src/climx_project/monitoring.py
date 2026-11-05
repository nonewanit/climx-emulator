"""Experiment monitoring (W&B).

One builder used by all runners: builds a Lightning-compatible logger from the
experiment config's `logger` section. Design goals:

- **Works without a W&B account**: if `WANDB_API_KEY` is missing (and
  `WANDB_MODE` is unset), runs go offline automatically — metrics land in the
  local `wandb/` directory and can be synced later with `wandb sync wandb/`.
- **Zero config for teammates with an account**: `wandb login` once, then all
  runs appear under the configured project/entity.

Config shape (per experiment):
    logger:
      backend: wandb            # none | wandb
      project: climx-lite
      name: ffnn                # optional; defaults to the experiment name
      offline: false            # optional; force offline mode
"""

import logging
import os
from typing import Any

from . import paths

logger = logging.getLogger(__name__)


def build_logger(cfg_logger: Any):
    """Build the logger object for a runner, or None when backend is 'none'.

    `cfg_logger` is the composed OmegaConf/DictConfig `logger` section (may be
    missing entirely). Returns a Lightning WandbLogger or None.
    """
    if cfg_logger is None:
        return None
    backend = getattr(cfg_logger, "backend", "none")
    if backend in (None, "none"):
        return None
    if backend != "wandb":
        raise ValueError(f"Unknown logger backend: {backend!r}")

    import wandb
    from lightning.pytorch.loggers import WandbLogger

    # Colleagues without a W&B account can still train: force offline mode so
    # wandb stores runs locally instead of failing on a missing API key.
    force_offline = bool(getattr(cfg_logger, "offline", False))
    if force_offline or (not os.environ.get("WANDB_API_KEY") and "WANDB_MODE" not in os.environ):
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.info("W&B: no API key found — running in offline mode (wandb/ dir).")

    name = getattr(cfg_logger, "name", None)
    project = getattr(cfg_logger, "project", "climx-lite")
    # Hydra chdirs into per-run dirs; keep W&B artifacts in one stable,
    # gitignored location (results/runs/wandb/) instead of scattering them.
    save_dir = str(paths.RUNS_DIR)
    return WandbLogger(project=project, name=name, save_dir=save_dir)


def log_scalar(wandb_logger, key: str, value: float) -> None:
    """Log a scalar to the run (no-op when the logger is None)."""
    if wandb_logger is not None:
        wandb_logger.experiment.log({key: value})


def finish(wandb_logger) -> None:
    """Close the W&B run (no-op when the logger is None)."""
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
