# ClimX — course project pipeline (Tsinghua ML 80245013, Fall 2026)

Code for the [ClimX](https://www.kaggle.com/competitions/climx-main-track) Kaggle
competition: extreme-aware emulation of the NorESM2-MM Earth System Model.
Proposal lives next door in [`../proposal/`](../proposal/). Kaggle closes
**Dec 15, 2026** (treat **Nov 30** as the safe final-submission target).

## Layout — how "many approaches" are organized

- `train.py` — the **single entry point**. An approach is chosen by config:
  `python train.py experiment=<name>`.
- `configs/` — one YAML per approach (`configs/experiment/*.yaml`), composed by
  Hydra with shared defaults in `configs/config.yaml` + `configs/data/lite.yaml`.
  CLI overrides work anywhere: `python train.py experiment=ffnn experiment.trainer.max_epochs=1`.
- `src/climx_project/` — shared plumbing, written **once** and reused by every
  approach: data download/mapping, preprocessing wrappers, the local validation
  harness (official `rnNSE.py` scorer), submission parquet builder/verifier.
  `src/climx_project/runners/` holds the thin per-model fit glue; everything
  after fit is shared in `runners/base.py`.
- `third_party/climx/` — vendored snapshot of the organizers' starter kit
  ([IPL-UV/ClimX](https://github.com/IPL-UV/ClimX) @ `21e6502`). **Read-only** —
  never edit; a needed patch gets documented here instead.
- `data/`, `results/` — gitignored artifacts. `results/scores.md` is the
  tracked score log (the mid-term report's data source).

**Adding the week-10 U-Net** = create `configs/experiment/unet.yaml` +
`src/climx_project/runners/unet.py` (copy `runners/ffnn.py`). Nothing else changes.

## Setup

```bash
mamba env create -f environment.yml     # env name: climx (CPU-only, aarch64-safe)
conda activate climx
pip install -e .                       # installs ONLY climx_project
```

Import rule: every script starts with `import climx_project` — that is the one
place the vendored starter kit gets put on `sys.path` (its code mixes relative
and `from src...` imports, so its root must be importable as a top-level
`src` package). Run scripts from the repo root; never `cd third_party/climx`.

## Manual Kaggle steps (one-time, before the first submission)

1. Every member: Kaggle account + accept the rules at
   https://www.kaggle.com/competitions/climx-main-track/rules
2. One member: create an API key (Account → Settings → Create New API Token),
   save it as `~/.kaggle/kaggle.json` (`chmod 600`).
3. One member creates the team; everyone joins **before the first submission**
   (merging teams later discards one team's submission history).

## Pipeline

```bash
make data          # download the Kaggle bundle + map it into data/lite/ (~800 MB)
make preprocess    # climatology + scaling params + X_train/y_train/X_test (.nc)
make climatology   # baseline 1 (also runs the ~1.0 scoring self-check)
make lps           # baseline 2
make ffnn          # baseline 3 (5 Lightning epochs, ~30-90 min CPU)
make verify FILE=results/submissions/submission_ffnn.parquet
make submit FILE=results/submissions/submission_ffnn.parquet MSG="FFNN baseline, lite"
```

Each run writes its predictions (zarr), indices (.nc), submission parquet, and a
row in `results/scores.md` (local nNSE on the SSP3-7.0 2081–2100 holdout).
Expected local-score band: −0.3 … −0.7, with climatology the worst.

## Local validation mirrors the leaderboard

The official scorer `third_party/climx/rnNSE.py` embeds the AR6 region mask and
the 192×288 grid — no external files. We hold out SSP3-7.0 2081–2100 (real
targets), compute GT indices from the raw daily fields with the official index
code, upscale both sides 12×18 → 192×288 (nearest + edge extrapolation — the
edge NaNs would otherwise invalidate the submission), and score with
`rnNSE.score()`. Organizer reference scores on the full test set: climatology
−0.5653, LPS −0.4487, FFNN −0.4664.

## Data routes

- Primary: Kaggle CLI (`make data`) — needs the API key above.
- Fallback: `python -m climx_project.data download-hf` (downloads `lite/` from
  the Hugging Face mirror `hf-mirror.com`; huggingface.co is blocked on this
  network, as is github.com).

## Known starter-kit bugs we work around

- `data_processing.py` looks for `scaling_params_lite.json`; the correct file is
  `scaling_parameters_lite.json` (preprocessing writes it). Never run
  `data_processing.py` verbatim.
- `train.py` has stale `n_forcing_vars=12`; the real value after the aerosol
  triple-split is 22 (set in `configs/experiment/ffnn.yaml`).
- The starter's parquet writer silently zero-fills NaNs — our submission module
  refuses to write instead.
