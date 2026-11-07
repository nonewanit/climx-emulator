# Full-Grid U-Net Plan (Track C — weeks 11–12)

Status: groundwork in progress (Nov 7). Launch after the mid-term report.

## Why

Lite (12×18) is saturated: U-Net 20-epoch −0.4340 local ≈ 5-epoch −0.4380, and
LB gains will be equally marginal — 216 pixels give convolutions no spatial
structure to exploit. On the full 192×288 grid the U-Net should decisively
beat the per-pixel FFNN (organizer full-grid refs: FFNN −0.4664). That is the
Phase B payoff.

## The core problem: 172 GB can't be preprocessed like lite

The starter's preprocess materializes X (22 channels) — at full resolution
that's 22 × 94k days × 55,296 cells × 4B ≈ 458 GB. Instead, full-grid training
must **stream**: keep everything dask-chunked from the HF zarr, apply the
preprocessing transforms lazily per chunk, and feed the trainer without ever
materializing the whole dataset. Same reversal machinery at inference.

## Where it runs: Colab (free T4, or the ~$100 rented-GPU budget)

- Colab can reach huggingface.co (this WSL cannot). Fallback: `hf-mirror.com`.
- Data access: the full tier is zarr (`isp-uv-es/ClimX`, `full/` folder) —
  open with `xarray.open_zarr` / `open_mfdataset(engine='zarr')`, chunks along
  time. No download; read-through with a small cache.
- 16 GB T4 fits a small U-Net at 192×288 (input 22×192×288 ≈ 5 MB; batch 8–16).
- The reproducibility bar (≤1 modern GPU, ≤24 h inference) is satisfied by
  design; fix seeds.

## Work items

1. **Streaming data path** (do now, on lite, so it's debugged before Colab):
   - `UNetEmulator.fit`: replace the in-memory numpy path with a dask-chunked
     `Dataset` (mirroring the starter's `NNDailyIterableDataset`), reading
     `X_train.nc`/`y_train.nc` in time chunks instead of `.to_numpy()`.
   - Keep `max_days` smoke mode working against the same code path.
2. **Grid handling**: the model already pads to a configurable `pad_to`;
   full grid uses `pad_to: [192, 288]` (no-op) — plus verify conv/pool parity
   (192×288 → 96×144 → 48×72 → 24×36: all even, clean).
   **Skip connections: ON for the full grid** (`skip_connections: true`). The
   lite A/B (2026-11-07) showed skips hurt on the coarse grid (−0.4348 vs
   −0.4289), but fine spatial detail exists only at 192×288 — classic U-Net
   evidence applies there. Report the ablation honestly.
3. **Preprocessing for the full tier on Colab**: run the starter's
   `preprocess_train` lazily against the zarr (writes `climatology_full.nc` +
   scaling params — small), then stream-serve the anomalies. Climatology
   computation is a one-time Colab CPU job (est. hours; cache the outputs).
4. **Colab notebook** (`notebooks/colab-full-unet.ipynb`): clone repo, pip deps
   (torch preinstalled), HF_ENDPOINT fallback, stream + train, save checkpoint,
   build submission via the shared tail.
5. **Submission**: full-grid indices need no upscale (already 192×288); the
   `submission.build_submission` path handles it (upscale is a no-op when the
   grid matches — verify).

## Risks

- HF throttling from Colab (mitigate: resume + small cache + regional subset
  for iteration).
- T4 memory at batch ≥16 (mitigate: batch 8, mixed precision via Lightning).
- Preprocessing compute time on Colab CPU (mitigate: run once, save
  `climatology_full.nc` + scaling JSON to Drive/HF, reuse across sessions).

## Acceptance

Full-grid U-Net beats the organizer FFNN reference (−0.4664) on the LB, i.e.
improves on our lite U-Net's −0.5437 by a decisive margin.
