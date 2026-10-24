"""Shared experiment tail: every runner ends with run_experiment().

Fit is model-specific (each runner does it); everything after fit is identical
for all approaches: evaluate on the test forcings, build + verify the Kaggle
submission parquet, run the SSP370 holdout validation with the official
scorer, and append the score to results/scores.md.
"""

import logging

from .. import paths, submission, validation

logger = logging.getLogger(__name__)


def run_experiment(emulator, model_name: str, dataset, X_train, y_train,
                   metadata: dict, X_test) -> dict:
    # LPS predict() cannot handle a size-1 forcing_scenario dim; harmless
    # squeeze for the others (NN/climatology handle it themselves).
    if "forcing_scenario" in X_test.dims and X_test.sizes["forcing_scenario"] == 1:
        X_test = X_test.isel(forcing_scenario=0, drop=True)

    logger.info("[%s] Evaluating on test forcings (SSP2-4.5) ...", model_name)
    y_pred, indices_pred = emulator.evaluate(
        X_test,
        metadata,
        results_path=paths.INDICES_DIR / f"{model_name}_evaluation.json",
        predictions_path=paths.PREDICTIONS_DIR / f"{model_name}_predictions.zarr",
        historical_data_path=dataset.hist,
        TARGET_VARIABLES=paths.TARGET_VARIABLES,
        LOAD_PREDICTIONS=False,
        compute_indices=True,
    )

    logger.info("[%s] Building Kaggle submission ...", model_name)
    parquet = submission.build_submission(indices_pred, model_name)
    verify = submission.verify_parquet(parquet)

    logger.info("[%s] Holdout validation (SSP3-7.0 2081-2100) ...", model_name)
    X_hold, y_raw = validation.make_holdout_split(X_train, y_train, dataset)
    local_score = validation.run_holdout_validation(
        emulator, model_name, X_hold, y_raw, metadata, dataset.hist
    )

    validation.append_score(model_name, local_score, parquet)
    return {"model": model_name, "local_nNSE": local_score,
            "parquet": str(parquet), "parquet_report": verify}
