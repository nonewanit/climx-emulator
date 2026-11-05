# One command per pipeline stage. Run everything from the repo root with the
# `climx` conda env active.

.PHONY: data preprocess climatology lps ffnn all verify submit

# Download the Kaggle bundle and map it into data/lite/
data:
	@python -m climx_project.data download
	@python -m climx_project.data map-lite
	@python -m climx_project.data check

# Preprocess (train + test). ~10-30 min CPU.
preprocess:
	@python -m climx_project.preprocess

climatology:
	@python train.py experiment=climatology

lps:
	@python train.py experiment=lps

ffnn:
	@python train.py experiment=ffnn

all: climatology lps ffnn

# make verify FILE=results/submissions/submission_ffnn.parquet
verify:
	@python -m climx_project.submission verify $(FILE)

# make submit FILE=results/submissions/submission_ffnn.parquet MSG="FFNN baseline"
submit:
	@python -m climx_project.submission submit $(FILE) "$(MSG)"

# Copy local TensorBoard logs into the tracked monitoring/ dir for teammates.
sync-tb:
	@rsync -a --delete results/runs/tensorboard/ monitoring/tensorboard/ 2>/dev/null || cp -r results/runs/tensorboard/* monitoring/tensorboard/
	@echo "synced -> monitoring/tensorboard/ (commit + push to share)"
