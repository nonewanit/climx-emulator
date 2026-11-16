"""Standalone holdout validation for saved models.

Re-scores a trained model against the SSP3-7.0 2081-2100 holdout with the
official scorer — no retraining. Every `train.py` run already does this at
the end (printing "Local nNSE ..." and appending results/scores.md); this
CLI re-checks any saved checkpoint, e.g. a best-epoch snapshot pulled from
W&B artifacts.

Usage:
    python -m climx_project.holdout unet
    python -m climx_project.holdout unet \\
        --checkpoint results/models/unet/best-epoch=01-val_loss=0.4000.ckpt
    python -m climx_project.holdout worldmodel \\
        --checkpoint results/models/worldmodel/best-epoch=01.ckpt
"""

import argparse
import logging

import climx_project  # noqa: F401  (vendor + env + compat activation)

from climx_project import data, paths, preprocess, validation
from climx_project.models.unet import UNetEmulator

logger = logging.getLogger(__name__)


def _load_emulator(model: str, ckpt_path, feature_vars):
    if model == "unet":
        if ckpt_path is not None:
            return UNetEmulator.from_checkpoint(ckpt_path,
                                                feature_vars=feature_vars)
        em = UNetEmulator({})
        em.load(paths.MODELS_DIR / "unet" / "unet.ckpt")
        return em
    if model == "worldmodel":
        from climx_project.models.worldmodel import WorldModelEmulator
        if ckpt_path is not None:
            return WorldModelEmulator.from_checkpoint(ckpt_path)
        em = WorldModelEmulator({})
        em.load(paths.MODELS_DIR / "worldmodel" / "worldmodel.ckpt")
        return em
    raise SystemExit(f"unknown model {model!r} (supported: unet, worldmodel)")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("unet", "worldmodel"),
                    help="which saved model to score")
    ap.add_argument("--checkpoint", default=None,
                    help="Lightning .ckpt path; default = the run's saved "
                         "results/models/<model>/<model>.ckpt (best-val weights)")
    ap.add_argument("--submission", action="store_true",
                    help="also run the TEST-set evaluate + build/verify the "
                         "Kaggle submission parquet (recovers the tail of a "
                         "crashed run without retraining)")
    args = ap.parse_args(argv)

    paths.ensure_dirs()
    dataset = data.open_datasets()
    pre = preprocess.load_preprocessed()
    X_train, y_train, metadata = pre["X_train"], pre["y_train"], pre["metadata"]
    X_test = pre["X_test"]

    emulator = _load_emulator(args.model, args.checkpoint,
                              feature_vars=list(X_train.data_vars))

    if args.submission:
        from climx_project import submission
        logger.info("Rebuilding the Kaggle submission from %s ...",
                    args.checkpoint or "the saved model")
        y_pred, indices_pred = emulator.evaluate(
            X_test, metadata,
            results_path=paths.INDICES_DIR / f"{args.model}_evaluation.json",
            predictions_path=paths.PREDICTIONS_DIR / f"{args.model}_predictions.zarr",
            historical_data_path=dataset.hist,
            TARGET_VARIABLES=paths.TARGET_VARIABLES,
            LOAD_PREDICTIONS=False, compute_indices=True)
        parquet = submission.build_submission(indices_pred, args.model)
        report = submission.verify_parquet(parquet)
        logger.info("submission parquet: %s (%s)", parquet, report)

    X_hold, y_raw = validation.make_holdout_split(X_train, y_train, dataset)
    score = validation.run_holdout_validation(
        emulator, args.model, X_hold, y_raw, metadata, dataset.hist)
    logger.info("LOCAL nNSE (%s): %.6f", args.model, score)
    print(f"LOCAL nNSE ({args.model}): {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
