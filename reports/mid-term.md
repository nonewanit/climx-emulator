# Mid-Term Report — ClimX: Extreme-Aware Emulation of NorESM2-MM

> Skeleton — fill in prose, keep the numbers below in sync with `../results/scores.md`.
> NeurIPS format required by the course; the proposal (`../proposal/proposal/main.tex`) is the LaTeX template.

**Team:** None Wanitchollakit (2026280062), Nattapol Kiatchaipipat (2026280075), Charoonroj Amornpativet (2026280056)
**Course:** Machine Learning 80245013, Tsinghua University, Fall 2026
**Due:** ~Nov 15, 2026

## 1. Introduction

- 1-2 paragraphs: ESM emulation motivation + the ClimX task (forcings → 7 daily fields on 192×288; scored on 15 extreme indices via region-wise nNSE over the held-out SSP2-4.5). Reuse the proposal's framing ([`../proposal/proposal/main.tex`](../proposal/proposal/main.tex)).
- Milestone goal (weeks 7–9, proposal Phase A): lite-data pipeline, baseline reproduction, first Kaggle submission.

## 2. Methods

- Pipeline (our repo, `climx-emulator`): Hydra configs + shared plumbing + one runner per approach; official scorer `rnNSE.py` embedded in local validation (AR6 mask, 192×288 upscaling).
- Data: lite bundle (~800 MB, 12×18 grid, 16× coarsened), official split (train = historical + SSP1-2.6/3-7.0/5-8.5; test = SSP2-4.5 forcings only).
- Three baselines: climatology, LPS (linear pattern scaling), FFNN (per-pixel MLP, 5 epochs, Lightning).

## 3. Results

Local holdout = SSP3-7.0, 2081–2100 (20 years), scored with the official metric; leaderboard scores from Kaggle:

| Model | Local nNSE (holdout) | Kaggle LB | Organizer full-grid reference |
|---|---|---|---|
| climatology | −0.6460 | *(pending — submit when quota allows)* | −0.5653 |
| LPS | −0.4409 | **−0.54241** | −0.4487 |
| FFNN | −0.4247 | **−0.55664** | −0.4664 |

- Self-check: climatology-vs-itself nNSE = **1.0** (scoring plumbing verified).
- Ordering matches the organizers' references; our LPS lands within 0.01 of their full-grid score.
- *(add the Kaggle leaderboard numbers once submissions score)*

## 4. Engineering findings (the interesting part)

Four real bugs found and fixed during the milestone — good methodology material:

1. **dask threads scheduler deadlock** in `xarray.to_netcdf` (all threads in `futex_do_wait`) → forced single-threaded scheduler globally.
2. **xarray + pandas 2.2/pyarrow `strftime` crash** (`ArrowStringArray` has no `.reshape`) → broke every percentile-based index (WSDI/CSDI/R95p); fixed by compat shim.
3. **Upstream stationarization-reversal bug**: January's climatology added to every day (~2 K annual bias, ~15–20 K summer cold) — fixed by restoring the daily-interpolation intent; round-trip bias now ~1e-5 K. Invalidated our first LPS score (−0.622 → −0.441 after the fix).
4. **Starter `pixel_batch` leak** into `pl.Trainer(**kwargs)`.

Also: GitHub/HF/Weights-&-Biases network blocks → mirror routes (`ghfast.top`, `hf-mirror` + `curl -4`, TensorBoard local dashboard as the default logger).

## 5. Next steps (weeks 10–12, Phase B)

- U-Net encoder–decoder (`configs/experiment/unet.yaml`, trained on Kaggle's free GPU with the lite data; full-grid on Colab after).
- World-model extension (temporal conditioning + autoregressive rollout), ensembles.

## 6. References

- Competition: https://www.kaggle.com/competitions/climx-main-track · starter kit vendored in `third_party/climx/`
- Proposal bibliography: [`../proposal/proposal/main.tex`](../proposal/proposal/main.tex)
