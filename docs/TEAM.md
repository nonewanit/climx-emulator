# Team Collaboration Guide

3-person team: None Wanitchollakit (2026280062), Nattapol Kiatchaipipat (2026280075), Charoonroj Amornpativet (2026280056).

## Access (one-time setup)

1. **GitHub** — the repo owner adds the other two members as collaborators:
   repo → Settings → Collaborators → Add people (write access). Everyone then
   works with their own GitHub account and commit identity.
2. **Kaggle** — ONE team for the whole group, created on the competition's
   Team page; everyone joins **before any further submissions** (changing
   teams later wipes submission history — this already happened once on
   Nov 6 and cost us the LPS/FFNN leaderboard entries).

## Everyday workflow

```bash
git pull                        # ALWAYS before starting work
# ... do your task ...
git add -A && git commit -m "what and why"
git push
```

- Work on `main` directly; keep commits small and descriptive. If two people
  touch the same file, the second `git pull` may conflict — resolve together.
- Never commit `data/` or `results/` artifacts (gitignored). The ONE tracked
  result file is `results/scores.md` — always update it when you get a new
  score, and pull before you run so you don't clobber someone else's row.

## How to run things (see README for full setup)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate climx
python train.py experiment=unet   # or climatology / lps / ffnn
```

- Local CPU: only smoke runs (`experiment.trainer.max_epochs=1
  experiment.trainer.max_days=120`).
- Real training = **Kaggle GPU, interactive notebook** (`notebooks/kaggle-unet.ipynb`,
  imported from GitHub — see `docs/EXPERIMENTS.md` for the 3-minute flow).
- **Headless `kaggle kernels push` runs are deprecated** (Nov 7): their
  sessions cannot be stopped remotely, logs are blind until the run ends, and
  outputs must be pulled. The `kaggle-headless/` and `kaggle-fullsmoke/`
  folders are kept for reference only — do not use them.
- Tracking = **W&B (cloud)**: set `WANDB_API_KEY` as a Kaggle notebook Secret
  and run with `logger.backend=wandb logger.project=climx`. The run link is
  shared in the team chat; everyone watches the same project page live.
  (None's campus network blocks wandb.ai directly — he opens it in a browser
  via VPN. Local CPU runs keep `logger.backend=tensorboard`.)

## Division of labor (agreed Nov 7)

| Who | Role |
|---|---|
| **Nattapol + Charoonroj** | **Experiment runners**: techniques + hyperparameter tuning on Kaggle's GPU, following [`docs/EXPERIMENTS.md`](EXPERIMENTS.md). No report writing, no code changes — if a knob is missing, ask None to add it. |
| **None (+ Claude Code)** | Reports (mid-term + final), the repo, and the core training from the proposal (U-Net main model, world-model phase C, ensembles). |

Runners own the "try it and record it" loop; None owns the "make it possible"
loop. Every number in the report must be reproducible by the person who ran it.

## Score log etiquette (`results/scores.md`)

One row per (model, run). Columns: date | model | local nNSE (SSP370 holdout) |
parquet | kaggle LB | commit. Fill the LB cell after the leaderboard scores.
When re-running a model, add a NEW row (history is evidence for the report).
