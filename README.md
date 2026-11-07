# ClimX — Extreme-Aware Emulation of NorESM2-MM

Course project for Tsinghua ML 80245013 (Fall 2026): join the
[ClimX](https://www.kaggle.com/competitions/climx-main-track) Kaggle competition
and build a fast ML emulator of the NorESM2-MM Earth System Model. The emulator
maps climate **forcings** (CO₂, aerosols) to **daily climate fields**; the
leaderboard scores how well the predicted fields reproduce 15 extreme-climate
indices (heatwaves, droughts, …) on a held-out future scenario (SSP2-4.5).

- Proposal + background reading: [`../proposal/`](../proposal/)
- Kaggle closes **Dec 15, 2026** — treat **Nov 30** as the safe final-submission target
- Mid-term report ~Nov 15, poster week 15, final report + code ~Jan 8, 2027

## Current results (week 7–9 milestone)

Local nNSE on our SSP3-7.0 2081–2100 holdout, scored with the official metric:

| Model | Local nNSE | Organizer full-grid reference |
|---|---|---|
| climatology | −0.6460 | −0.5653 |
| LPS | −0.4409 | −0.4487 |
| FFNN | **−0.4247** | −0.4664 |

Every run appends to [`results/scores.md`](results/scores.md) (the tracked score log).
nNSE ∈ (−1, 1]: 1 = perfect, 0 = "predict the mean", negative = worse than that.

## Quickstart (from zero to a score)

```bash
# 1. One-time: install Miniforge (https://github.com/conda-forge/miniforge) and
#    clone the repo, then create the environment (~10 min):
git clone <repo-url> && cd climx
mamba env create -f environment.yml
conda activate climx
pip install -e .
```

```bash
# 2. One-time, every team member: Kaggle account + accept the competition rules
#    (https://www.kaggle.com/competitions/climx-main-track/rules). ONE member
#    also creates an API key (Account → Settings → Create New API Token) saved as
#    ~/.kaggle/access_token (CLI v2.x; legacy name kaggle.json), chmod 600, and
#    creates the team — everyone joins
#    BEFORE the first submission (merging teams later wipes submission history).
```

```bash
# 3. Download the lite data (~800 MB) and preprocess it (~10–30 min CPU):
make data
make preprocess
```

```bash
# 4. Run an experiment (each ends with a local nNSE + submission parquet):
python train.py experiment=climatology    # ~10 min
python train.py experiment=lps            # ~10 min
python train.py experiment=ffnn           # ~30–90 min (5 Lightning epochs, CPU)
# any config value is overridable on the CLI:
python train.py experiment=ffnn experiment.trainer.max_epochs=10
```

```bash
# 5. Verify and submit (needs the Kaggle API key from step 2):
python -m climx_project.submission verify results/submissions/submission_ffnn.parquet
python -m climx_project.submission submit results/submissions/submission_ffnn.parquet "FFNN baseline"
# then add the leaderboard score to results/scores.md (kaggle LB column)
```

## Monitoring runs

The FFNN experiment logs loss curves through Lightning, with two backends behind
one config switch (`configs/experiment/ffnn.yaml` → `logger.backend`):

- **`tensorboard` (default) — local browser dashboard, no account, no network:**
  ```bash
  tensorboard --logdir results/runs/tensorboard     # then open http://localhost:6006
  ```
  Shows train/val loss curves, hyperparameters, and the final local nNSE per run.
- **`wandb` — cloud dashboard** (nice when your network can reach wandb.ai):
  `wandb login`, then runs appear under the project `climx-lite`. Without an
  account (or from a blocked network) it auto-falls-back to offline mode — runs
  land in `results/runs/wandb/` and can be synced later with
  `wandb sync results/runs/wandb`.
- `logger.backend: none` disables monitoring.

The builder is shared: [`src/climx_project/monitoring.py`](src/climx_project/monitoring.py).

**Smoke runs** (don't clobber real artifacts):
```bash
python train.py experiment=ffnn experiment.trainer.max_epochs=1 \
  experiment.record=false experiment.model_dir=results/models/ffnn_smoke
```

## Adding a new approach (weeks 10+: U-Net, ConvLSTM, ensembles)

The repo is organized so a new model = **two new files, nothing else**:

1. `configs/experiment/<name>.yaml` — model hyperparams, trainer settings,
   optional `logger:` block. Copy `ffnn.yaml`.
2. `src/climx_project/runners/<name>.py` — construct the model, fit it, then
   call `base.run_experiment(...)`. Copy `runners/ffnn.py`.

Run it with `python train.py experiment=<name>`. Data loading, preprocessing,
holdout validation (official scorer), submission building, and score logging
are all shared — you never touch them.

## Repo layout

```
train.py                 # single Hydra entry point: python train.py experiment=<name>
configs/                 # one YAML per approach (+ shared data/defaults)
src/climx_project/       # shared plumbing: data, preprocess, validation (official
                         #   rnNSE scorer), submission, W&B monitoring, runners/
third_party/climx/       # vendored organizers' starter kit @ 21e6502 (read-only;
                         #   bug fixes live in src/climx_project/compat.py)
notebooks/               # exploration (playground_lite.ipynb)
data/                    # gitignored: downloads + preprocessing outputs
results/                 # gitignored artifacts + TRACKED scores.md
reports/                 # mid-term / final report drafts (professor deliverables)
```

## Local validation mirrors the leaderboard

The official scorer (`third_party/climx/rnNSE.py`) embeds the AR6 land-region
mask and the 192×288 grid. We hold out the last 20 years of a training scenario
(SSP3-7.0, 2081–2100 — the test scenario has no released targets), compute the
ground-truth indices from the raw daily fields with the official index code,
upscale predictions 12×18 → 192×288 (nearest + edge extrapolation), and score
with `rnNSE.score()`. The climatology run also performs a **self-check** that
must equal ~1.0 — it proves the scoring plumbing is correct.

## Troubleshooting

- **github.com / huggingface.co blocked** (network restrictions): clone via
  `https://ghfast.top/https://github.com/...`; datasets via
  `python -m climx_project.data download-hf` (hf-mirror.com, `curl -4`).
- **`HDF5` / dask hangs**: already handled globally by
  `src/climx_project/__init__.py` (single-threaded dask, HDF5 file locking off).
- **NaN in a submission parquet**: our pipeline refuses to write it (the
  starter's writer would silently zero-fill). Fix the source, don't bypass.
- More detail for contributors: [CLAUDE.md](CLAUDE.md).

## License

MIT — see [LICENSE](LICENSE). The vendored starter kit in `third_party/climx/`
keeps its own MIT license (IPL). Open-sourcing code + weights (MIT/Apache-2.0)
is also a competition requirement for prize eligibility.
