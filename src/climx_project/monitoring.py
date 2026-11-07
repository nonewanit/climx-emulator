"""Experiment monitoring (W&B and TensorBoard).

One builder used by all runners: builds a Lightning-compatible logger from the
experiment config's `logger` section.

Config shape (per experiment):
    logger:
      backend: wandb | tensorboard | none
      project: climx-lite        # W&B only
      entity: null               # W&B only; team/account override (else wandb default)
      name: ffnn                  # optional; defaults to the experiment name
      offline: false              # W&B only; force offline mode

Backend notes:
- **wandb** — cloud dashboard. Auto-offline when `WANDB_API_KEY` is missing
  (or the network can't reach wandb.ai); runs land in `results/runs/wandb/`
  and can be synced later with `wandb sync results/runs/wandb`.
- **tensorboard** — fully local browser dashboard; launch with
  `tensorboard --logdir results/runs/tensorboard` → http://localhost:6006.
"""

import logging
import os
from typing import Any

from . import paths

logger = logging.getLogger(__name__)


def build_logger(cfg_logger: Any):
    """Build the logger object for a runner, or None when backend is 'none'.

    `cfg_logger` is the composed OmegaConf/DictConfig `logger` section (may be
    missing entirely). Returns a Lightning logger (WandbLogger or
    TensorBoardLogger) or None.
    """
    if cfg_logger is None:
        return None
    backend = getattr(cfg_logger, "backend", "none")
    if backend in (None, "none"):
        return None

    name = getattr(cfg_logger, "name", None)

    if backend == "wandb":
        return _build_wandb(cfg_logger, name)
    if backend == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger(
            save_dir=str(paths.RUNS_DIR / "tensorboard"),
            name=name or "experiment",
        )
    raise ValueError(f"Unknown logger backend: {backend!r}")


def _build_wandb(cfg_logger: Any, name):
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    # Colleagues without a W&B account can still train: force offline mode so
    # wandb stores runs locally instead of failing on a missing API key.
    force_offline = bool(getattr(cfg_logger, "offline", False))
    if force_offline or (not os.environ.get("WANDB_API_KEY") and "WANDB_MODE" not in os.environ):
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.info("W&B: no API key found — running in offline mode (wandb/ dir).")

    project = getattr(cfg_logger, "project", "climx-lite")
    entity = getattr(cfg_logger, "entity", None) or None
    # Hydra chdirs into per-run dirs; keep W&B artifacts in one stable,
    # gitignored location (results/runs/wandb/) instead of scattering them.
    save_dir = str(paths.RUNS_DIR)
    return WandbLogger(project=project, name=name, entity=entity, save_dir=save_dir)


def log_scalar(logger_obj, key: str, value: float, step: int = 0) -> None:
    """Log a scalar to the run — backend-agnostic (no-op when logger is None)."""
    if logger_obj is None:
        return
    exp = logger_obj.experiment
    if hasattr(exp, "add_scalar"):  # TensorBoard
        exp.add_scalar(key, value, global_step=step)
    elif hasattr(exp, "log"):  # W&B
        exp.log({key: value})


def finish(logger_obj) -> None:
    """Close the run (no-op when the logger is None)."""
    if logger_obj is None:
        return
    exp = logger_obj.experiment
    if hasattr(exp, "finish"):
        exp.finish()
