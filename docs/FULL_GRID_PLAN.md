# Full-Grid U-Net Plan (Track C — weeks 11–12)

Status: **SMOKE PASSED (Nov 8)** — `FULLRES_SMOKE_OK`. The full-res code path
is proven end to end; the real run is now an engineering scaling problem, not
a feasibility question.

**UPDATE (Nov 9): the full tier is available as a LOCAL Kaggle dataset**
(`dtdoww/climx-full-resolution-data`, 171 GB — all 7 files incl. train_ssp585).
Training now reads the local mount (~1 GB/s) instead of HF streaming; the
private-cache-dataset plan is unnecessary for Phase B (`scripts/fullres_train.py`
writes a subsampled fp16 cache to /tmp per session, ~15 min). The
`scripts/full_cache.py` job stays for Phase C (contiguous ×1 cache).

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
per chunk.

Streaming is the real cost, though: the smoke preprocessed ~15 GB of remote
reads in ~2 h → **~2–3 MB/s effective** (single-threaded dask, per-task file
opens, range-request latency). Naive remote streaming of 234 GB/epoch ≈
**25–30 h/epoch** — infeasible. The fix: a ONE-TIME preprocessing job writes
a compacted cache — **every day, float16 + zstd (~3× total: X ≈ 78 GB,
y ≈ 25 GB)** — uploaded as **private Kaggle datasets** (split across two:
X-cache and y-cache). Training attaches them as local input mounts
(~1 GB/s, no streaming) — epochs become GPU-bound again (~1 h).

Subsampling happens at TRAINING TIME, not in the cache: Phase B's Dataset
reads every 3rd sample (16k samples/epoch) — data volume 3× smaller, each
sample still a full daily field. NEVER weekly-average: the LB indices are
daily and extreme-aware (tx90p, consecutive-day runs, r99p) — weekly means
erase the extremes the score measures. The ×1 cache also serves **Phase C
directly**: the world model reads all days, contiguous, for next-DAY
prediction — one cache, both phases, nothing rebuilt.

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

1. **One-time cache job** (`scripts/full_cache.py`) — DONE, running: streams
   the real `hist` + full train (2015–2100, 2 scenarios), writes fp16+zstd
   scenario-pure zarr chunks to /tmp, auto-uploads ~20 GB batches as private
   per-scenario datasets, manifest-resumable across 9 h sessions.
2. **Chunked training Dataset** — DONE (`_CacheDataset` + `_fit_cached` in
   `unet.py`, integration-tested): block-buffered reads over the cache,
   subsample-at-load (`sample_step=3`), shared trainer plumbing with the
   in-memory lite path. Local mount → I/O is a non-issue; GPU-bound epochs.
3. **Throughput**: batch 16–32 + AMP (`16-mixed`; T4 fp16 tensor cores ≈
   65 TFLOPS vs 8 fp32) — the smoke's 14 samples/s should become ~40–80,
   i.e. ~20–35 min/epoch at 16k samples.
4. **Checkpoint/resume across 9 h sessions**: `pl.Trainer` checkpoint every
   epoch, resume next session.
5. **Real training notebook** (`notebooks/kaggle-full-unet.ipynb`): smoke
   skeleton + cache attachment + checkpoint-resume.
6. **Grid handling**: `pad_to: [192, 288]` (no-op) — parity verified in the
   smoke. **Skips ON for the full grid** (lite A/B: skips hurt only on the
   coarse grid). Model width is a Hydra knob (`base_channels`), see below.
7. **Submission**: no upscale needed at 192×288 (verify `build_submission`).

## Session budget (estimate, Kaggle 30 h GPU/week)

| Step | Sessions |
|---|---|
| 1. Cache job (preprocess + upload, 39 GB) | ~2 (one-time) |
| 2. Training: 5–10 epochs × ~0.5–1 h (batch 16–32, AMP), EarlyStopping patience 3 | ~1–2 |
| 3. Predict 7 305 test days + indices + submission | ~0.5 |

Total ≈ **4–5 sessions** (~35 h of the 30 h weekly quota → spans 2 weeks or
use both accounts). Every step checkpoints, so nothing is lost to the 9 h cap.
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
