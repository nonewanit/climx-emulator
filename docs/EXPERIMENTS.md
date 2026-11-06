# Experiment Menu — for the runners

Your job: try techniques and hyperparameters, run them on Kaggle's GPU, record
every result in `results/scores.md`. Nobody writes report prose except None.
If you want a NEW knob that isn't in this menu, ask None first — it may need a
small code change (5 minutes for him), not a workaround.

## The 3-minute workflow (no code, no local setup)

1. `git pull` in your clone of the repo (ask None for access).
2. Edit **`kaggle-headless/params.json`** — change only the values.
3. Push it to Kaggle's GPU:

```bash
cd climx/kaggle-headless
kaggle kernels push -p .          # needs YOUR Kaggle API key in ~/.kaggle/kaggle.json
```

4. Watch progress:
   - `kaggle kernels status nonewanitchollakit/climx-unet-headless`
   - or the notebook page (you can open it from Kaggle → your notebooks).
5. When complete: `kaggle kernels output -p results/kaggle nonewanitchollakit/climx-unet-headless`
   → `results/kaggle/outputs/scores.md` has the new row; copy it into the
   repo's `results/scores.md` (new row, don't overwrite), commit, push.
6. The submission parquet is in `results/kaggle/outputs/` — the Kaggle
   leaderboard has a 2-submissions/day quota shared by the WHOLE TEAM:
   coordinate before submitting.

## Current menu (pick one, mark it in the table below)

| # | What | params.json change | Why it matters |
|---|---|---|---|
| 1 | Longer training | `experiment.trainer.max_epochs=30` | Early stopping caps it; more epochs = better fit |
| 2 | Capacity up | `experiment.model.model_params.base_channels=128` | Bigger U-Net, still small |
| 3 | Capacity down | `...base_channels=32` | Overfitting check — if 32 ≈ 64, we're capacity-bound |
| 4 | LR sweep | `experiment.model.model_params.learning_rate=0.0003` (and `0.003`) | 1e-3 may be suboptimal |
| 5 | Batch size | `experiment.trainer.batch_size=32` (and `128`) | BN statistics + convergence speed |
| 6 | Loss weights | `experiment.model.model_params.variable_weights='[1,2,0.5,0.5,2,2,2]'` | Emphasize the index-driving vars (tas*/pr) even more |
| 7 | Seeds | `experiment.seed=7` (and `123`) | Ensemble material for Phase D |

Baseline to beat: **U-Net local nNSE −0.4380 / LB −0.54374** (5 epochs, seed 42).

## Rules

- One row per run in `scores.md` — never overwrite someone else's row.
- `git pull` immediately before editing params.json and before committing.
- If a run errors, paste the error into the team chat — None fixes the repo
  and pushes; you just re-run your params.
- If the leaderboard quota is the blocker, queue the parquet and say so in
  `scores.md`'s LB column ("queued").
