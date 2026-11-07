# Experiment Menu — for the runners

Your job: try techniques and hyperparameters, run them on Kaggle's GPU, record
every result in `results/scores.md`. Nobody writes report prose except None.
If you want a NEW knob that isn't in this menu, ask None first — it may need a
small code change (5 minutes for him), not a workaround.

## The 3-minute workflow (interactive notebook + W&B, no local setup)

1. On Kaggle: **File → Import Notebook → GitHub** → repo
   `nonewanit/climx-emulator` → `notebooks/kaggle-unet.ipynb` → Import.
2. Right panel → **Accelerator → GPU** (T4 x2 or P100).
3. (One-time) Add-ons → **Secrets** → add `WANDB_API_KEY` (from
   https://wandb.ai/settings → API keys). Without it, training still runs —
   W&B just logs offline locally.
4. Run the notebook. For experiment knobs, add CLI overrides to the train
   cell, e.g.:

   ```bash
   !python train.py experiment=unet \
       experiment.seed=7 experiment.trainer.max_epochs=5 \
       logger.backend=wandb logger.project=climx logger.name=unet-seed7
   ```

5. **Watch live:** the W&B run link prints in the notebook output; the whole
   team watches the shared project page (https://wandb.ai/... — None's campus
   network blocks wandb, he views it in a browser via VPN).
6. When training ends, **Stop Session** (top-right → Session Options → Stop
   Session). Interactive sessions stop cleanly — unlike the old headless
   pushes, which could not be stopped remotely at all.
7. The notebook prints `results/scores.md` at the end — copy your NEW row into
   your clone of the repo, commit, push. (If the run errored, paste the error
   into the team chat; None fixes the repo and you just re-run.)

## Current menu

**STATUS (Nov 7): the lite grid is SATURATED.** Single-model hyperparameter
sweeps on lite are dead GPU hours (5 epochs −0.4380 vs 20 epochs −0.4340
local — the ceiling is the 216-pixel grid, not the training). The ONE lite
activity still worth GPU time is **Phase D: seeds for the ensemble** — the
rules allow an unweighted mean of index maps across realizations of the SAME
emulator, which reliably beats any single seed on the LB.

### Ensemble run plan (3 runners, 3 notebook runs)

1. Runner A: `experiment.seed=7`
2. Runner B: `experiment.seed=123`
3. Runner C: `experiment.seed=42` (re-run for a consistent trio)

Same config except the seed — nothing else may differ (that's the
"same emulator" rule). Each run produces `results/submissions/submission_unet.parquet`;
download it from the notebook's working dir (right panel → Output → Files) and
rename (`submission_unet_seed7.parquet`, ...). Then:

```bash
python scripts/ensemble_indices.py \
  results/submissions/submission_unet_seed7.parquet \
  results/submissions/submission_unet_seed42.parquet \
  results/submissions/submission_unet_seed123.parquet \
  -o results/submissions/submission_ensemble.parquet
```

One ensemble parquet = one Kaggle submission slot (2/day team quota — coordinate!).

### (Suspended) single-model lite sweeps — only if the ensemble fails to beat LPS

| # | What | train.py override |
|---|---|---|
| 1 | Capacity up | `experiment.model.model_params.base_channels=128` |
| 2 | LR sweep | `experiment.model.model_params.learning_rate=0.0003` (and `0.003`) |
| 3 | Loss weights | `experiment.model.model_params.variable_weights='[1,2,0.5,0.5,2,2,2]'` |

Baseline to beat: **U-Net local nNSE −0.4380 / LB −0.54374** (5 epochs, seed 42);
LPS holds the top LB spot at −0.54241.

## Rules

- One row per run in `scores.md` — never overwrite someone else's row.
- `git pull` immediately before a run and before committing.
- If a run errors, paste the error into the team chat — None fixes the repo
  and pushes; you just re-run your params.
- If the leaderboard quota is the blocker, queue the parquet and say so in
  `scores.md`'s LB column ("queued").
