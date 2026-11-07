"""FFNN (per-pixel MLP) baseline runner: the starter's NNBaseline trained with
PyTorch Lightning on the lite grid. This runner is the template future
approaches (U-Net in weeks 10+) will be copied from.

Config extras this runner understands (all optional, in the experiment config):
    logger: {backend: wandb, project: ..., name: ...}   # W&B monitoring
    record: false        # smoke runs: skip submission + score-log writes
    model_dir: <path>    # override the checkpoint directory
"""

import logging
from pathlib import Path

from .. import monitoring, paths
from . import base

logger = logging.getLogger(__name__)


def run(cfg, dataset, X_train, y_train, metadata, X_test) -> dict:
    from src.models.nn_model import NNBaseline

    base.set_seed(cfg.experiment.seed)

    model_params = dict(cfg.experiment.model.model_params)
    emulator = NNBaseline(model_params)

    trainer_params = dict(cfg.experiment.trainer)
    model_dir = Path(cfg.experiment.get("model_dir") or str(paths.MODELS_DIR / "ffnn"))
    trainer_params["model_path"] = str(model_dir)

    wandb_logger = monitoring.build_logger(getattr(cfg.experiment, "logger", None))
    if wandb_logger is not None:
        trainer_params["logger"] = wandb_logger

    emulator.fit(X_train, y_train, trainer_params=trainer_params)

    model_dir.mkdir(parents=True, exist_ok=True)
    emulator.save(model_dir / "ffnn.ckpt")

    record = bool(cfg.experiment.get("record", True))
    result = base.run_experiment(emulator, "ffnn", dataset, X_train, y_train,
                                 metadata, X_test, record=record)

    monitoring.log_scalar(wandb_logger, "local_nNSE", result["local_nNSE"])
    monitoring.finish(wandb_logger)
    return result
