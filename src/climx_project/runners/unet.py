"""U-Net runner (Phase B): encoder-decoder conv network on forcing stacks.

Same shape as runners/ffnn.py: construct -> fit -> shared tail. Train on the
lite grid locally for smoke checks, on Kaggle's free GPU for real runs, and on
the full 192x288 grid (Colab, weeks 11-12) by changing `pad_to` and the data
tier — the model class is identical.
"""

import logging
from pathlib import Path

from .. import monitoring, paths
from ..models.unet import UNetEmulator
from . import base

logger = logging.getLogger(__name__)


def run(cfg, dataset, X_train, y_train, metadata, X_test) -> dict:
    base.set_seed(cfg.experiment.seed)

    model_params = dict(cfg.experiment.model.model_params)
    emulator = UNetEmulator(model_params)

    trainer_params = dict(cfg.experiment.trainer)
    model_dir = Path(cfg.experiment.get("model_dir") or str(paths.MODELS_DIR / "unet"))
    trainer_params["model_path"] = str(model_dir)

    wandb_logger = monitoring.build_logger(getattr(cfg.experiment, "logger", None))
    if wandb_logger is not None:
        trainer_params["logger"] = wandb_logger

    emulator.fit(X_train, y_train, trainer_params=trainer_params)
    emulator.save(model_dir / "unet.ckpt")

    record = bool(cfg.experiment.get("record", True))
    result = base.run_experiment(emulator, "unet", dataset, X_train, y_train,
                                 metadata, X_test, record=record)

    monitoring.log_scalar(wandb_logger, "local_nNSE", result["local_nNSE"])
    monitoring.finish(wandb_logger)
    return result
