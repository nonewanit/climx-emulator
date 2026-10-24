"""Single entry point for all ClimX experiments (Hydra).

Usage:
    python train.py experiment=climatology
    python train.py experiment=lps
    python train.py experiment=ffnn
    python train.py experiment=ffnn trainer.max_epochs=1      # CLI override
    python train.py -m experiment=climatology,lps,ffnn        # multirun

Import order matters: `import climx_project` must run at module import time —
it activates the vendored starter kit on sys.path BEFORE Hydra chdirs into
the per-run output directory. All paths are __file__-relative, so Hydra's
cwd change is otherwise harmless.
"""

import logging

import climx_project  # noqa: F401  (vendor activation; do not move below)

import hydra
from omegaconf import DictConfig

from climx_project import data as cdata
from climx_project import preprocess
from climx_project.paths import ensure_dirs

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ensure_dirs()

    dataset = cdata.open_datasets(chunks=cfg.data.raw_chunks)
    preprocessed = preprocess.load_preprocessed()

    # instantiate() calls the _target_ (the experiment's run() function) with
    # these positional/keyword arguments and returns its result directly.
    result = hydra.utils.instantiate(cfg.experiment.runner, cfg, dataset, **preprocessed)
    logger.info("Run finished: %s", result)


if __name__ == "__main__":
    main()
