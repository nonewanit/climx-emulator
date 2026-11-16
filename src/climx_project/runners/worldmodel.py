"""World-model runner (Phase C): temporal-conditioned U-Net + AR rollout.

Same shape as runners/unet.py: construct -> fit -> shared tail. The model
lives in models/worldmodel.py; its predict() rolls out autoregressively from
the learned missing-history prior, so the starter's evaluate() — and hence
the whole shared tail (reversal, indices, holdout scoring, submission) —
works unchanged.

Checkpoints land in results/models/worldmodel/: best-val + periodic epoch
snapshots (Lightning ModelCheckpoint) plus the final worldmodel.ckpt. With
logger.backend=wandb and log_model=all, every checkpoint is also uploaded as
a W&B artifact to the run.
"""

import logging
from pathlib import Path

from .. import monitoring, paths
from ..models.worldmodel import WorldModelEmulator
from . import base

logger = logging.getLogger(__name__)


def run(cfg, dataset, X_train, y_train, metadata, X_test) -> dict:
    base.set_seed(cfg.experiment.seed)

    model_params = dict(cfg.experiment.model.model_params)
    emulator = WorldModelEmulator(model_params)

    trainer_params = dict(cfg.experiment.trainer)
    model_dir = Path(cfg.experiment.get("model_dir") or str(paths.MODELS_DIR / "worldmodel"))
    trainer_params["model_path"] = str(model_dir)

    wandb_logger = monitoring.build_logger(getattr(cfg.experiment, "logger", None))
    if wandb_logger is not None:
        trainer_params["logger"] = wandb_logger

    emulator.fit(X_train, y_train, trainer_params=trainer_params)
    emulator.save(model_dir / "worldmodel.ckpt")

    record = bool(cfg.experiment.get("record", True))
    result = base.run_experiment(emulator, "worldmodel", dataset, X_train, y_train,
                                 metadata, X_test, record=record)

    monitoring.log_scalar(wandb_logger, "local_nNSE", result["local_nNSE"])
    monitoring.finish(wandb_logger)
    return result
