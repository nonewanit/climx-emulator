# Full-Grid U-Net Plan (Track C — weeks 11–12)

Status: **SMOKE PASSED (Nov 8)** — `FULLRES_SMOKE_OK`. The full-res code path
is proven end to end; the real run is now an engineering scaling problem, not
a feasibility question.

## Smoke verdict (2026-11-08, Kaggle T4 x2, interactive notebook)

All stages green on the 192×288 grid: remote HF streaming (time-chunked,
single-threaded dask, scenarios concatenated along `forcing_scenario`),
`daily_forcings`, full-grid `preprocess_train`/`preprocess_test`, 1-epoch fit
(skips ON, `devices=1`), predict, and the complete preprocessing reversal
(`compute_indices=False` — the smoke slice can't satisfy the 1961–1990 base
period). Hard-won config facts:

- **`devices=1` is REQUIRED**: `accelerator="auto"` on the 2-GPU box launches
  DDP and hangs 1801 s waiting for the second client (`DistStoreError`).
- **Single-threaded dask only**: the threads scheduler races seeks on the
  shared remote file handle (v5's hours-long thrash).
- `ds.hist = ds.train` must be re-applied after `daily_forcings` rebinds
  `ds.train` (the starter's climatology reads `ds.hist`).
- HF 503 bursts happen; `huggingface_hub` retries and recovers.

## Timing measurements (from the smoke)

| Stage | Smoke size | Measured | Real size | Projection |
|---|---|---|---|---|
| preprocess | 800 days, climatology from train slice | ~2–3 h | 94k days hist + train, real climatology | ~8–12 h streaming |
| fit | 60 days (15 batches, batch 4) | 55 min = **3.7 min/batch** | ~48k days ≈ 12k batches/epoch | ~30 h/epoch at batch 16 → **unacceptable as-is** |
| predict | 30 days | 12.7 s = 0.42 s/day | 7 305 test days | ~51 min (fine) |

The fit projection kills the naive "same code, more data" plan. The real run
needs (a) **chunked in-memory-free training data** and (b) **throughput work**
(bigger batches, mixed precision, fewer-parameter variants, maybe
downsampled-input training) plus **checkpointing across 9 h sessions** — see
work items.

## The core problem: 172 GB can't be preprocessed like lite

The starter's preprocess materializes X (22 channels) — at full resolution
that's 22 × 94k days × 55,296 cells × 4B ≈ 458 GB. Kaggle's working disk
(~20 GB) can't hold even `X_train.nc`. Full-grid training must **stream**:
keep everything dask-chunked from the HF NetCDFs, apply the preprocessing
transforms lazily per chunk, and feed the trainer without ever materializing
the whole dataset. Same reversal machinery at inference.

## Where it runs: Kaggle interactive notebooks (proven) — Colab as fallback

- Kaggle T4 x2: HF streaming + full pipeline proven by the smoke. 30 h GPU
  quota/week, **9 h session cap** → training must checkpoint and resume
  across sessions (`/kaggle/working` persists between sessions of the same
  notebook; keep the preprocessing outputs + model checkpoints there).
- Colab (free T4 / rented-GPU budget) remains an option; same code path.
- HF_ENDPOINT=mirror fallback for networks that can't reach huggingface.co.

## Work items (ordered)

1. **Chunked training data** (blocks everything): replace `UNetEmulator.fit`'s
   `to_numpy` (loads ~13 GB even for the smoke slice) with a dask-chunked
   map-style `Dataset` reading time-chunks of `X_train.nc`/`y_train.nc` (or
   lazily preprocessed chunks) per batch — mirroring the starter's
   `NNDailyIterableDataset`. Keep `max_days` smoke mode on the same path.
2. **Throughput**: batch 8–16 + Lightning mixed precision (16-mixed) first
   (~2×), then model slimming (base_channels 32, or stride-2 convs instead of
   MaxPool) if an epoch still exceeds a session. Target: ≥1 epoch/session.
3. **Real preprocessing**: run `preprocess_train` against the real `hist`
   (63 GB stream) + full train once, cache `climatology_full.nc` + scaling
   JSON in `/kaggle/working` (small, persists). Climatology is one-time;
   do NOT write full X/y to disk.
4. **Real training notebook** (`notebooks/kaggle-full-unet.ipynb`): reuse the
   smoke notebook skeleton; add checkpoint-resume (`pl.Trainer` checkpoint
   loading) so a 9 h session ends with a saved model, not lost work.
5. **Grid handling**: `pad_to: [192, 288]` (no-op) — conv/pool parity verified
   in the smoke (192×288 → 96×144 → 48×72 → 24×36, all even).
   **Skip connections: ON for the full grid** (`skip_connections: true`). The
   lite A/B (2026-11-07) showed skips hurt on the coarse grid (−0.4348 vs
   −0.4289), but fine spatial detail exists only at 192×288 — classic U-Net
   evidence applies there. Report the ablation honestly.
6. **Submission**: full-grid indices need no upscale (already 192×288); the
   `submission.build_submission` path handles it (upscale is a no-op when the
   grid matches — verify).

## Risks

- HF throttling (503s; mitigate: built-in retries, resume, regional subset
  for iteration).
- T4 memory at batch ≥16 (mitigate: mixed precision, slimmer model).
- 9 h session cap mid-training (mitigate: checkpoint every epoch, resume in
  the next session).
- Epoch wall-clock (mitigate: work item 2 before committing to a full run).

## Acceptance

Full-grid U-Net beats the organizer FFNN reference (−0.4664) on the LB, i.e.
improves on our lite U-Net's −0.5437 by a decisive margin.
