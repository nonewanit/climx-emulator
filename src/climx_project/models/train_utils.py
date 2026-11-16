"""Shared Lightning trainer plumbing for the learned models (unet, worldmodel).

Every learned approach gets the SAME checkpoint policy, in one place:

- best-val checkpoint (monitor="val_loss", keep 1 + last), FIRST in the
  callback list because `trainer.checkpoint_callback` returns the first;
- periodic epoch snapshots (epoch-05.ckpt, epoch-10.ckpt, ...) so a crash or
  an over-eager early stop never loses the whole run;
- EarlyStopping on val_loss so max_epochs can be set generously.

After fit, the module is reloaded from the best-val checkpoint — val loss
fluctuates at this scale and last != best.
"""

import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Trainer-param keys that belong to the surrounding fit() (dataset/loader
# construction), not to pl.Trainer. Kept in sync with unet.py's fit()/
# _fit_cached() and worldmodel.py's fit().
NON_TRAINER_KEYS = ("batch_size", "num_workers", "model_path", "max_days",
                    "early_stop_patience", "checkpoint_every", "scenarios",
                    "sample_step", "block_days", "cache_blocks",
                    "metric_weights")


def in_notebook_kernel() -> bool:
    """True when running inside an active Jupyter kernel (widgets possible)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in getattr(shell, "config", {})
    except Exception:
        return False


def is_real_terminal() -> bool:
    """True only for a real interactive terminal.

    Notebook `!python ...` subprocesses (Kaggle/Colab/papermill) run under a
    PTY, so sys.stdout.isatty() is True there — yet their captured output
    renders every \r-updated bar refresh as a NEW line (the val-bar spam).
    Jupyter marks such subprocesses with JPY_PARENT_PID in the inherited
    environment; Kaggle additionally sets KAGGLE_KERNEL_RUN_TYPE, Colab
    COLAB_*, papermill PAPERMILL_MILL.
    """
    if not sys.stdout.isatty():
        return False
    markers = ("JPY_PARENT_PID", "KAGGLE_KERNEL_RUN_TYPE",
               "PAPERMILL_MILL", "COLAB_JUPYTER_IP")
    return not any(k in os.environ for k in markers)


class ProgressLog(pl.Callback):
    """One log line per epoch — progress visibility where the bars are
    disabled (notebook subprocesses). No \r rewrites, so no capture spam."""

    def __init__(self):
        import time
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, *_):
        import time
        metrics = trainer.callback_metrics
        ep = trainer.current_epoch + 1
        tr = metrics.get("train_loss_epoch")
        va = metrics.get("val_loss")
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        dt_min = (time.time() - self._t0) / 60 / ep
        logger.info("EPOCH %d/%d | train %.4f val %.4f | lr %.1e | %.1f min/ep",
                    ep, trainer.max_epochs,
                    float(tr) if tr is not None else float("nan"),
                    float(va) if va is not None else float("nan"),
                    lr, dt_min)


def run_trainer(model: pl.LightningModule, load_fn, train_dl, val_dl, tp: dict):
    """Fit `model` with the shared callback policy; return it with best-val weights.

    `load_fn(ckpt_path) -> module` must reconstruct the module class from a
    Lightning checkpoint (e.g. UNetModule.load_from_checkpoint), so the best
    checkpoint can be reloaded after fit.
    """
    trainer_kwargs = {k: v for k, v in tp.items() if k not in NON_TRAINER_KEYS}
    trainer_kwargs.setdefault("logger", False)
    # Lightning's TQDMProgressBar rewrites its line with \r on every refresh;
    # a notebook's captured `!python train.py` output renders each refresh as
    # a NEW line (hundreds of val-bar lines per epoch). In a real terminal
    # \r works, and inside a Jupyter KERNEL (%run) Lightning renders
    # notebook-safe bars — keep them. Anywhere else (notebook subprocess —
    # detected by env markers, since Kaggle gives it a pty — or redirected
    # logs), disable the bars and add the per-epoch ProgressLog line. An
    # explicit `trainer.enable_progress_bar=true|false` config value wins
    # (setdefault).
    bars_unsafe = not in_notebook_kernel() and not is_real_terminal()
    if bars_unsafe:
        trainer_kwargs.setdefault("enable_progress_bar", False)
    model_path = Path(tp.get("model_path", "results/models/model"))
    model_path.mkdir(parents=True, exist_ok=True)
    trainer_kwargs.setdefault("default_root_dir", str(model_path))

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = trainer_kwargs.get("callbacks", [])
    # KEEP THIS FIRST: trainer.checkpoint_callback returns the first in the
    # list, and the post-fit load uses it for the best-val weights.
    callbacks.append(ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
        dirpath=str(model_path), filename="best-{epoch:02d}-{val_loss:.4f}",
    ))
    # Periodic snapshots so a crash or an over-eager early stop never loses
    # the whole run.
    checkpoint_every = int(tp.get("checkpoint_every", 5))
    if checkpoint_every > 0:
        callbacks.append(ModelCheckpoint(
            every_n_epochs=checkpoint_every, save_top_k=-1, save_last=False,
            dirpath=str(model_path), filename="epoch-{epoch:02d}",
        ))
    # Self-regulating long runs: stop when validation loss plateaus, so
    # max_epochs can be set generously without wasting GPU time.
    callbacks.append(EarlyStopping(
        monitor="val_loss", mode="min",
        patience=int(tp.get("early_stop_patience", 3)), min_delta=1e-4,
        verbose=True, check_on_train_epoch_end=False,
    ))
    # Where the bars are disabled, one clean progress line per epoch instead.
    if bars_unsafe:
        callbacks.append(ProgressLog())
    trainer_kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dl, val_dl)

    # Evaluate with the BEST-val weights, not the last epoch's.
    best = getattr(trainer.checkpoint_callback, "best_model_path", None)
    if best:
        model = load_fn(best)
        logger.info("Loaded best-val checkpoint for evaluation: %s", best)
    return model
