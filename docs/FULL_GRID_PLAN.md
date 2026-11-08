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
| preprocess | 800 days, climatology from train slice | ~2–3 h | 94k days hist + train, real climatology | ~8–12 h streaming (one-time) |
| fit — GPU epoch | 60 days × 2 scenarios = 30 batches | **8 s (3.53 it/s)** | ~48k samples | **~1 h/epoch** — GPU is NOT the bottleneck |
| fit — array materialization | 800 days ≈ 10 GB | ~55 min (to_numpy of the lazy graph) | 234 GB | **impossible in RAM or on the 20 GB disk** |
| predict | 30 days | 12.7 s = 0.42 s/day | 7 305 test days | ~51 min (fine) |

The 55-min fit stage was dominated by `to_numpy` materializing the training
arrays, NOT by training (the epoch itself took 8 s). The real run therefore
has exactly one hard problem: **234 GB of training arrays can't be
materialized** — RAM and Kaggle's ~20 GB working disk both rule it out. The
data must be **streamed chunk-by-chunk** with the preprocessing transforms
(climatology/scaling/stationarization, cached as small files) applied lazily
per chunk. At disk speed (~1 GB/s) that's ~4 min of I/O per epoch — training
a real epoch fits comfortably inside a 9 h session, and even multi-epoch runs
work with checkpointing.

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

1. **Streaming training data** (blocks everything): replace `UNetEmulator.fit`'s
   `to_numpy` (materializes ~10 GB even for the smoke slice — 55 of the
   55 min fit stage) with a dask-chunked map-style `Dataset` that reads
   time-chunks per batch — mirroring the starter's `NNDailyIterableDataset`.
   The GPU epoch is ~8 s per 30 batches, so even a simple chunked loader wins.
   Keep `max_days` smoke mode on the same path.
2. **Lazy preprocessing per chunk** (the 234 GB problem): cache
   `climatology_full.nc` + scaling JSON (small, one-time ~8–12 h job), then
   apply climatology/scaling/stationarization on the fly per time-chunk
   during training. Never write full X/y to disk — it doesn't fit.
3. **Throughput, only if needed**: the GPU does ~14 samples/s at batch 4
   (3.53 it/s) — a real epoch ≈ 1 h already. Mixed precision (16-mixed) and
   batch 16–32 are cheap wins on top; model slimming is optional.
4. **Checkpoint/resume across 9 h sessions**: `pl.Trainer` checkpoint every
   epoch, resume in the next session (skips re-preprocessing via idempotency
   like the smoke script).
5. **Real training notebook** (`notebooks/kaggle-full-unet.ipynb`): reuse the
   smoke notebook skeleton; add checkpoint-resume so a 9 h session ends with
   a saved model, not lost work.
6. **Grid handling**: `pad_to: [192, 288]` (no-op) — conv/pool parity verified
   in the smoke (192×288 → 96×144 → 48×72 → 24×36, all even).
   **Skip connections: ON for the full grid** (`skip_connections: true`). The
   lite A/B (2026-11-07) showed skips hurt on the coarse grid (−0.4348 vs
   −0.4289), but fine spatial detail exists only at 192×288 — classic U-Net
   evidence applies there. Report the ablation honestly.
7. **Submission**: full-grid indices need no upscale (already 192×288); the
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
