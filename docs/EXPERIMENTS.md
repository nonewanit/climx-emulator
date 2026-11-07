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
kaggle kernels push -p .          # needs YOUR Kaggle API key in ~/.kaggle/access_token (CLI v2.x)
```

4. Watch progress:
   - `kaggle kernels status nonewanitchollakit/climx-unet-headless`
   - or the notebook page (you can open it from Kaggle → your notebooks).
5. When complete, pull ONLY the outputs (seconds, not 20 minutes — the
   `--file-pattern` skips the bulky data/ and the repo clone):
   `kaggle kernels output -p results/kaggle nonewanitchollakit/climx-unet-headless --file-pattern '^outputs/'`
   → `results/kaggle/outputs/scores.md` has the new row; copy it into the
   repo's `results/scores.md` (new row, don't overwrite), commit, push.
6. The submission parquet is in `results/kaggle/outputs/` — the Kaggle
   leaderboard has a 2-submissions/day quota shared by the WHOLE TEAM:
   coordinate before submitting.

## Current menu

**STATUS (Nov 7): the lite grid is SATURATED.** Single-model hyperparameter
sweeps on lite are dead GPU hours (5 epochs −0.4380 vs 20 epochs −0.4340
local — the ceiling is the 216-pixel grid, not the training). The ONE lite
activity still worth GPU time is **Phase D: seeds for the ensemble** — the
rules allow an unweighted mean of index maps across realizations of the SAME
emulator, which reliably beats any single seed on the LB.

### Ensemble run plan (3 runners, 3 pushes)

1. Runner A: `params.json` with `"experiment.trainer.max_epochs": 5`, `"experiment.seed": 7`
2. Runner B: same but `"experiment.seed": 123`
3. Runner C: same but `"experiment.seed": 42` (re-run for a consistent trio)

Same config except the seed — nothing else may differ (that's the
"same emulator" rule). Each push's `outputs/submission_unet.parquet` gets
pulled and renamed (`submission_unet_seed7.parquet`, ...). Then:

```bash
python scripts/ensemble_indices.py \
  results/kaggle/outputs/submission_unet_seed7.parquet \
  results/kaggle/outputs/submission_unet_seed42.parquet \
  results/kaggle/outputs/submission_unet_seed123.parquet \
  -o results/submissions/submission_ensemble.parquet
```

One ensemble parquet = one Kaggle submission slot (2/day team quota — coordinate!).

### (Suspended) single-model lite sweeps — only if the ensemble fails to beat LPS

| # | What | params.json change |
|---|---|---|
| 1 | Capacity up | `experiment.model.model_params.base_channels=128` |
| 2 | LR sweep | `experiment.model.model_params.learning_rate=0.0003` (and `0.003`) |
| 3 | Loss weights | `experiment.model.model_params.variable_weights='[1,2,0.5,0.5,2,2,2]'` |

Baseline to beat: **U-Net local nNSE −0.4380 / LB −0.54374** (5 epochs, seed 42);
LPS holds the top LB spot at −0.54241.

## Rules

- One row per run in `scores.md` — never overwrite someone else's row.
- `git pull` immediately before editing params.json and before committing.
- If a run errors, paste the error into the team chat — None fixes the repo
  and pushes; you just re-run your params.
- If the leaderboard quota is the blocker, queue the parquet and say so in
  `scores.md`'s LB column ("queued").
