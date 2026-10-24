"""FFNN (per-pixel MLP) baseline runner: the starter's NNBaseline trained with
PyTorch Lightning on the lite grid. This runner is the template future
approaches (U-Net in weeks 10+) will be copied from."""

import logging

import torch

from .. import paths
from . import base

logger = logging.getLogger(__name__)


def run(cfg, dataset, X_train, y_train, metadata, X_test) -> dict:
    from src.models.nn_model import NNBaseline

    torch.manual_seed(cfg.experiment.seed)

    model_params = dict(cfg.experiment.model.model_params)
    emulator = NNBaseline(model_params)

    trainer_params = dict(cfg.experiment.trainer)
    trainer_params["model_path"] = str(paths.MODELS_DIR / "ffnn")
    emulator.fit(X_train, y_train, trainer_params=trainer_params)

    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    emulator.save(paths.MODELS_DIR / "ffnn.ckpt")

    return base.run_experiment(emulator, "ffnn", dataset, X_train, y_train,
                               metadata, X_test)
