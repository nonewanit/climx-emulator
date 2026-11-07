# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Code for the **ClimX** Kaggle competition (extreme-aware emulation of the NorESM2-MM Earth System Model) — Tsinghua ML course 80245013, Fall 2026, weeks 7–9 milestone: lite-data pipeline, 3 of 4 baselines (climatology, LPS, FFNN; GNN deferred), local validation with the official scorer, first Kaggle submission. Proposal + research notes live in the sibling `../proposal/` (docs-only). Kaggle closes **Dec 15, 2026** (treat Nov 30 as the safe target); mid-term report ~Nov 15.

## Environment & commands

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate climx   # conda env `climx` (python 3.11, CPU)
pip install -e .          # installs ONLY climx_project; run from repo root
```

- `python train.py experiment=climatology|lps|ffnn` — run an approach (single Hydra entry point; CLI overrides like `experiment.trainer.max_epochs=1` work anywhere; `-m` for multirun).
- `make data | preprocess | climatology | lps | ffnn | verify FILE=... | submit FILE=... MSG=...` — pipeline stages.
- Module CLIs: `python -m climx_project.data {download,download-hf,map-lite,check}`, `python -m climx_project.preprocess`, `python -m climx_project.submission {verify,submit}`.
- No tests/linter; correctness is verified by the run artifacts (sizes, `results/scores.md`, parquet invariants).
- Hydra chdirs into a per-run dir — full logs land in `results/runs/<experiment>/<timestamp>/` (gitignored); stdout/stderr still goes to the console.

## Architecture

- **One approach = one Hydra config + one thin runner.** `train.py` composes `configs/config.yaml` + `configs/experiment/<name>.yaml`, instantiates the `runner._target_`, and the runner does model-specific fit then delegates to the shared tail. Adding the week-10 U-Net = new config + `src/climx_project/runners/unet.py` (copy `ffnn.py`); nothing else changes.
- **Shared plumbing lives once** in `src/climx_project/`: `data.py` (download/map/check), `preprocess.py` (starter preprocess_train/test wrappers; writes `data/preprocessing_data_lite/`), `validation.py` (SSP3-7.0 2081–2100 holdout, GT indices from raw fields, upscale 12×18→192×288, `rnNSE.score` — mirrors the leaderboard exactly; hard-asserts rows % 55296 to avoid rnNSE's silent non-regional fallback), `submission.py` (parquet build + read-back verification, refuses to let the starter's writer silently zero-fill NaNs), `monitoring.py` (shared logger builder: `backend: tensorboard` = local browser dashboard via `tensorboard --logdir results/runs/tensorboard`; `backend: wandb` = cloud, auto-offline when no key/blocked network). This machine's network blocks wandb.ai and wandb.cn — tensorboard is the default.
- `base.run_experiment(..., record=)` — `record=False` (smoke runs) skips submission parquet + scores.md writes; combine with `experiment.model_dir=` + `experiment.trainer.max_epochs=1` so smoke runs never clobber real artifacts.
- **`third_party/climx/` is a vendored snapshot** of IPL-UV/ClimX @ `21e6502` — never edit it, never `cd` into it. Its code mixes relative and `from src...` absolute imports with no `__init__.py`, so its root must sit on `sys.path`: `src/climx_project/__init__.py` is the ONE place that does this (plus `HDF5_USE_FILE_LOCKING=FALSE` and `dask.config.set(scheduler="single-threaded")`). **Every script must `import climx_project` first**, before any `from src...` import.
- `results/scores.md` is the tracked score log (date | model | local nNSE | parquet | Kaggle LB | commit) — the mid-term report's data source. Everything else under `data/` and `results/` is gitignored.

## Machine constraints (hard-won; do not "fix" without reason)

- WSL2 **aarch64**, **no GPU**, 7.5 GB RAM. Everything here is CPU/lite-scale; full-grid training belongs on Colab/Kaggle (weeks 10+).
- **Network**: `github.com` and `huggingface.co` are blocked. Clone git via `https://ghfast.top/https://github.com/...`; download HF data from `https://hf-mirror.com` with **`curl -4`** (this machine has no IPv6 route — `huggingface_hub`'s HTTP client hangs).
- **dask scheduler must stay single-threaded** — the threads scheduler deadlocked in `to_netcdf` (all threads in `futex_do_wait`), and `zarr` writes are at the same risk.
- `to_netcdf(..., compute=False)` returns a Delayed that **must be `.compute()`d** — discarding it writes only the file header (~300 KB files, no data).
- Kaggle CLI v2.2.5 credentials live at `~/.kaggle/access_token` (new-style; legacy `kaggle.json` also accepted — `climx_project.data` checks for either). Team must be registered **before the first submission** (merging teams later wipes history). Daily submission quota ~2.
- **Collaboration workflow (Nov 7 pivot)**: real training runs in INTERACTIVE Kaggle notebooks (`notebooks/kaggle-unet.ipynb` imported from GitHub) with W&B cloud tracking (`logger.backend=wandb`, `WANDB_API_KEY` as a notebook Secret; wandb.ai blocked on this machine → view in browser via VPN). Headless `kaggle kernels push` is deprecated — its sessions can't be stopped remotely and logs are blind. `kaggle-headless/` + `kaggle-fullsmoke/` are legacy. No CLI/API stop-session exists; only the notebook page's "Stop Session" ends a session.

## Starter-kit bugs we route around

- `data_processing.py` looks for the non-existent `scaling_params_lite.json` (correct: `scaling_parameters_lite.json`) — never run that file.
- `train.py` has stale `n_forcing_vars=12` (correct post-split: **22** — 5 GHG + 15 aerosol-split + 2 seasonal).
- Notebook defaults `DATA_VERSION='full'` + an organizer-internal `data_path` — the adapted copy is `notebooks/playground_lite.ipynb`.
- **Stationarization reversal is broken upstream**: `_stationarize_single_scenario` broadcasts the raw monthly climatology and `evaluate()` collapses it with `isel(month=0)` — January's climatology added to every day (~2 K annual bias, summer ~15–20 K cold). Fixed by monkeypatch in `src/climx_project/compat.py` (`_patch_stationarize_reverse`); the round-trip test (anomalies → reversal → raw fields) must stay at ~1e-5 K bias.

## Validation conventions & expectations

Local holdout = SSP3-7.0, 2081–2100 (matches the starter's own NN val convention). Expected local nNSE band **−0.3 … −0.7**, climatology worst (≈ −0.5; NOT 0 — nNSE=0 means "predict the period's own mean", and the 1850–2014 climatology under SSP370 warming is systematically off). Organizer full-test references: climatology −0.5653, LPS −0.4487, FFNN −0.4664. The climatology runner's self-check (~1.0) validates the scoring plumbing; if it fails, stop and debug before trusting any score.
