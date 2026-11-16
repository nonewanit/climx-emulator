"""U-Net climate emulator (Phase B main model).

A small encoder-decoder convolutional network mapping a stack of forcing
channels to the 7 target-variable fields, in SCALED-ANOMALY space (same target
space as the starter's baselines). Subclasses the starter's `Emulator` ABC, so
`evaluate()` — the preprocessing reversal, official index computation, and the
whole shared tail in `climx_project.runners.base` — works unchanged.

Grid handling: the lite grid is 12x18; inputs are padded to a clean power-of-2
size (16x32, configurable) for the 4x-downsampling path and cropped back to
the native grid on output. On the full 192x288 grid, use `pad_to: [192, 288]`
(no padding) — the same class works for both data tiers.
"""

import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader, Dataset

# Imported AFTER `import climx_project` (vendor sys.path activation).
from src.models.emulator import Emulator

logger = logging.getLogger(__name__)

TARGET_VARIABLE_ORDER = ["huss", "pr", "psl", "sfcWind", "tas", "tasmax", "tasmin"]
# The 15 leaderboard indices are built from tas/tasmax/tasmin/pr — weight those
# higher; huss/psl/sfcWind are part of the task but don't drive the score.
DEFAULT_VARIABLE_WEIGHTS = [1.0, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetModule(pl.LightningModule):
    def __init__(self, n_in=22, n_out=7, base_channels=64,
                 variable_weights=None, learning_rate=1e-3, pad_to=(16, 32),
                 skip_connections=True, cell_weights=None):
        super().__init__()
        self.save_hyperparameters()
        b = base_channels
        self.pad_h, self.pad_w = int(pad_to[0]), int(pad_to[1])
        self.skip = bool(skip_connections)
        self.variable_weights = (
            torch.tensor(variable_weights, dtype=torch.float32)
            if variable_weights is not None else torch.ones(n_out)
        )
        # Optional scorer-derived per-cell, per-variable loss weights
        # (n_out, H, W) on the NATIVE grid, from climx_project.metrics_weighting.
        # When set, they REPLACE the hand-tuned variable_weights in the loss.
        self.register_buffer("cell_weights",
                             None if cell_weights is None
                             else torch.as_tensor(cell_weights, dtype=torch.float32))

        self.enc1 = DoubleConv(n_in, b)
        self.enc2 = DoubleConv(b, 2 * b)
        self.bottleneck = DoubleConv(2 * b, 4 * b)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Classic U-Net skips: encoder features are concatenated into the
        # decoder (spatial sizes match because of the padded working grid).
        # skip_connections=False reproduces the plain encoder-decoder used in
        # the first lite submissions.
        if self.skip:
            self.dec1 = DoubleConv(4 * b + 2 * b, 2 * b)  # bottleneck + enc2
            self.dec2 = DoubleConv(2 * b + b, b)          # dec1 + enc1
        else:
            self.dec1 = DoubleConv(4 * b, 2 * b)
            self.dec2 = DoubleConv(2 * b, b)
        self.head = nn.Conv2d(b, n_out, 1)

    def forward(self, x):
        h_in, w_in = x.shape[-2:]
        # Pad to the working grid (e.g. 12x18 -> 16x32); crop back on output.
        pad_l = (self.pad_w - w_in) // 2
        pad_r = self.pad_w - w_in - pad_l
        pad_t = (self.pad_h - h_in) // 2
        pad_b = self.pad_h - h_in - pad_t
        x = nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b))

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d1_in = torch.cat([self.up1(b), e2], dim=1) if self.skip else self.up1(b)
        d1 = self.dec1(d1_in)
        d2_in = torch.cat([self.up2(d1), e1], dim=1) if self.skip else self.up2(d1)
        d2 = self.dec2(d2_in)
        out = self.head(d2)
        return out[:, :, :h_in, :w_in]

    def _step(self, batch, split):
        x, y = batch  # x: (B, n_in, H, W), y: (B, n_out, H, W)
        y_hat = self(x)
        if self.cell_weights is not None:
            # Metric-consistent loss (A/B arm B): scorer-derived per-cell,
            # per-variable weights, mean-normalized so the loss scale stays
            # comparable to the uniform-cell loss.
            loss = (self.cell_weights * (y_hat - y) ** 2).mean()
        else:
            weights = self.variable_weights.to(y_hat.device)
            loss = (weights.view(1, -1, 1, 1) * (y_hat - y) ** 2).mean()
        # sync_dist=True: under DDP each rank otherwise logs only its own
        # subset of the val batches, biasing early stopping and the best-ckpt
        # choice (no-op on a single device). Same fix as the world model.
        self.log(f"{split}_loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Constant-LR training plateaus well above what the model can reach
        # (observed: val improvements shrank to ~0.01/epoch at epoch ~21 and
        # the holdout nNSE stalled just below climatology). Halve the LR when
        # val_loss stalls for 4 epochs.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss",
                                 "interval": "epoch",
                                 "frequency": 1}}


class _ArrayDataset(Dataset):
    """In-memory map-style dataset over pre-stacked (n_samples, C, H, W) arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i]),
                torch.from_numpy(self.y[i]))


class _BlockSampler(torch.utils.data.Sampler):
    """Block-local sampling: shuffle the ORDER of 128-day blocks each epoch
    and serve the samples within each block together. A random batch of 16
    samples otherwise touches ~16 different blocks (~1.5 s each of zarr
    decompress+read) — block-local batches touch exactly one, so the loader
    stops dominating the training loop."""

    def __init__(self, samples, block_days: int):
        self.samples = samples
        self.block_days = block_days
        self._epoch = 0

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        from collections import defaultdict
        rng = np.random.default_rng(42 + self._epoch)
        self._epoch += 1
        blocks = defaultdict(list)
        for i, (scen, t) in enumerate(self.samples):
            blocks[(scen, t // self.block_days)].append(i)
        # Shuffle the keys as a plain Python list: numpy's permutation would
        # coerce the (str, int) tuples to STRING arrays (ints became strings)
        # and every lookup missed - with a defaultdict, the miss silently
        # created empty entries and the epoch had zero batches.
        keys = list(blocks.keys())
        rng.shuffle(keys)
        idxs = []
        for b in keys:
            rng.shuffle(blocks[b])
            idxs.extend(blocks[b])
        return iter(idxs)


class _CacheDataset(Dataset):
    """Map-style dataset over the fp16 scenario-pure cache (scripts/full_cache.py).

    Serves (scenario, day) samples without materializing anything: each zarr
    time-block is loaded once into a small LRU buffer and its days are served
    from memory. `step` subsamples days at LOAD time (daily physics kept —
    never averaged), so the same ×1 cache feeds Phase B (step=3) and Phase C
    (step=1, contiguous).
    """

    def __init__(self, cache_dir: Path, scenarios, day_indices, step: int = 1,
                 block_days: int = 128, cache_blocks: int = 12):
        self.block_days = block_days
        self.cache: dict = {}
        self.lru: list = []
        self.cache_blocks = cache_blocks
        self.x_paths = {}
        self.y_paths = {}
        self.time = {}
        # Open every store ONCE and keep the handles: re-opening 29 stores per
        # block load (xr.open_zarr metadata reads) made the training loop
        # crawl at ~50 s/batch. Direct zarr chunk reads are milliseconds.
        import zarr
        self.groups = {}
        for scen in scenarios:
            self.x_paths[scen] = sorted((cache_dir / f"X_{scen}").glob("*.zarr")) \
                if (cache_dir / f"X_{scen}").is_dir() \
                else sorted(cache_dir.glob(f"X_*_{scen}.zarr"))
            self.y_paths[scen] = sorted(cache_dir.glob(f"Y_*_{scen}.zarr")) or \
                sorted(cache_dir.glob(f"y_*_{scen}.zarr"))
            if not self.x_paths[scen] or not self.y_paths[scen]:
                raise FileNotFoundError(
                    f"No cache files for scenario {scen} under {cache_dir}")
            self.groups[scen] = {}
            for p in self.x_paths[scen] + self.y_paths[scen]:
                g = zarr.open_group(str(p), mode="r")
                var = next(k for k in g.array_keys() if g[k].ndim == 3)
                self.groups[scen][str(p)] = (g, var)
            self.time[scen] = xr.open_zarr(self.x_paths[scen][0]).time.values
        self.samples = [(scen, t) for scen in scenarios
                        for t in day_indices[scen][::step]]

    def __len__(self):
        return len(self.samples)

    def _load_block(self, scen, block):
        t0 = block * self.block_days
        t1 = min(t0 + self.block_days, len(self.time[scen]))
        tsel = slice(t0, t1)

        def read_var(p):
            g, var = self.groups[scen][str(p)]
            arr = g[var][tsel].astype(np.float32)
            if arr.shape[0] == 0:
                raise RuntimeError(f"{p.name}: empty read at time {tsel}")
            return arr

        xs = [read_var(p) for p in self.x_paths[scen]]
        ys = [read_var(p) for p in self.y_paths[scen]]
        # (C, T, H, W) -> (T, C, H, W)
        x_block = np.stack(xs, axis=1)
        y_block = np.stack(ys, axis=1)
        key = (scen, block)
        self.cache[key] = (x_block, y_block)
        self.lru.append(key)
        while len(self.lru) > self.cache_blocks:
            self.cache.pop(self.lru.pop(0), None)

    def __getitem__(self, i):
        scen, t = self.samples[i]
        block = t // self.block_days
        key = (scen, block)
        if key not in self.cache:
            self._load_block(scen, block)
        off = t - block * self.block_days
        x_block, y_block = self.cache[key]
        return (torch.from_numpy(x_block[off]),
                torch.from_numpy(y_block[off]))


class UNetEmulator(Emulator):
    def __init__(self, model_params):
        self.model_params = dict(model_params)
        self.model = UNetModule(**self.model_params)
        self.feature_vars = None
        self.target_vars = None

    def fit(self, X_train, y_train, trainer_params=None, **kwargs):
        # Cache-based dispatch: pass the cache DIR instead of datasets and the
        # chunked _CacheDataset path is used (full-res training).
        if isinstance(X_train, (str, Path)):
            return self._fit_cached(Path(X_train), trainer_params)
        self.feature_vars = list(X_train.data_vars)
        self.target_vars = list(y_train.data_vars)

        tp = dict(trainer_params or {})

        # A/B arm B: replace the hand-tuned variable_weights with
        # scorer-derived per-cell weights (land mask + 1/Var + region + index
        # counts). Rebuild the module so the weights are registered as a
        # buffer BEFORE fit (they then round-trip through the Lightning
        # checkpoints via save_hyperparameters).
        if tp.get("metric_weights"):
            from .. import metrics_weighting

            w = metrics_weighting.compute_metric_weights_for_fit(
                y_train, self.target_vars)
            model_path = Path(tp.get("model_path", "results/models/unet"))
            model_path.mkdir(parents=True, exist_ok=True)
            np.save(model_path / "metric_weights.npy", w)
            self.model_params = dict(self.model_params, cell_weights=w)
            self.model = UNetModule(**self.model_params)

        # Same split convention as the starter's NN (projections only, all
        # scenarios); val starts at 2081 to avoid the starter's overlapping
        # 2080 year.
        X_tr = X_train.sel(time=slice("2015-01-01", "2080-12-31"))
        y_tr = y_train.sel(time=slice("2015-01-01", "2080-12-31"))
        X_val = X_train.sel(time=slice("2081-01-01", "2100-12-31"))
        y_val = y_train.sel(time=slice("2081-01-01", "2100-12-31"))

        # Smoke runs (max_days > 0) use time-slices that usually don't reach
        # 2081-2100, so split by fraction of the available days instead of the
        # absolute-date convention.
        max_days = int(tp.get("max_days", 0) or 0)
        if max_days > 0:
            n = int(X_train.sizes["time"])
            cut = max(1, int(n * 0.8))
            logger.info("smoke split: %d days total, train %d / val %d (max_days=%d)",
                        n, cut, n - cut, max_days)
            X_tr = X_train.isel(time=slice(0, cut))
            y_tr = y_train.isel(time=slice(0, cut))
            X_val = X_train.isel(time=slice(cut, None))
            y_val = y_train.isel(time=slice(cut, None))
            X_tr = X_tr.isel(time=slice(0, max_days))
            y_tr = y_tr.isel(time=slice(0, max_days))
            X_val = X_val.isel(time=slice(0, max(1, max_days // 4)))
            y_val = y_val.isel(time=slice(0, max(1, max_days // 4)))

        def to_numpy(X_ds):
            stacked = X_ds.stack(sample=("forcing_scenario", "time"))
            arr = (stacked.to_array(dim="feature")
                   .transpose("feature", "sample", "lat", "lon")
                   .values.astype(np.float32))
            return arr.transpose(1, 0, 2, 3)  # (sample, feature, lat, lon)

        logger.info("Loading U-Net training arrays into memory ...")
        x_tr = to_numpy(X_tr)
        y_tr = to_numpy(y_tr)
        x_val = to_numpy(X_val)
        y_val = to_numpy(y_val)
        logger.info("train %s, val %s", x_tr.shape, x_val.shape)

        train_dl = DataLoader(_ArrayDataset(x_tr, y_tr),
                              batch_size=int(tp.get("batch_size", 64)),
                              shuffle=True,
                              num_workers=int(tp.get("num_workers", 0)))
        val_dl = DataLoader(_ArrayDataset(x_val, y_val),
                            batch_size=int(tp.get("batch_size", 64)),
                            shuffle=False,
                            num_workers=int(tp.get("num_workers", 0)))

        self._train(train_dl, val_dl, tp)

    def _train(self, train_dl, val_dl, tp):
        """Shared trainer plumbing: callback setup, fit, best-checkpoint load.

        Implementation lives in models/train_utils.py so the world model gets
        the identical checkpoint/early-stop policy.
        """
        from .train_utils import run_trainer

        self.model = run_trainer(self.model,
                                 lambda ckpt: UNetModule.load_from_checkpoint(ckpt),
                                 train_dl, val_dl, tp)

    def _fit_cached(self, cache_dir: Path, trainer_params):
        """Train from the fp16 scenario cache (scripts/full_cache.py output).

        Trainer-param extras (all filtered from pl.Trainer):
          scenarios:   ["ssp126", "ssp370"] — which cache scenarios to train on
          sample_step: 3 — subsample days at LOAD time (daily physics kept)
          max_days:    0 — smoke truncation of the per-scenario day list
          block_days / cache_blocks: zarr block-buffer sizing
        """
        tp = dict(trainer_params or {})
        scenarios = list(tp.get("scenarios", ["ssp126", "ssp370"]))
        step = int(tp.get("sample_step", 3))
        max_days = int(tp.get("max_days", 0) or 0)
        block_days = int(tp.get("block_days", 64))
        cache_blocks = int(tp.get("cache_blocks", 6))

        # Cache file names: X_<var>_<scenario>.zarr / y_<var>_<scenario>.zarr.
        self.feature_vars = [p.stem.split("_", 1)[1].rsplit("_", 1)[0]
                             for p in sorted(cache_dir.glob(f"X_*_{scenarios[0]}.zarr"))]
        self.target_vars = list(TARGET_VARIABLE_ORDER)
        logger.info("cache vars: X=%s", self.feature_vars)

        def day_index(scen, lo_year, hi_year):
            p = sorted(cache_dir.glob(f"X_*_{scen}.zarr"))[0]
            years = xr.open_zarr(p).time.dt.year.values
            return [i for i, y in enumerate(years) if lo_year <= y <= hi_year]

        tr_idx = {s: day_index(s, 2015, 2080) for s in scenarios}
        val_idx = {s: day_index(s, 2081, 2100) for s in scenarios}
        if max_days > 0:
            tr_idx = {s: v[:max_days] for s, v in tr_idx.items()}
            val_idx = {s: v[:max(1, max_days // 4)] for s, v in val_idx.items()}

        train_ds = _CacheDataset(cache_dir, scenarios, tr_idx, step=step,
                                 block_days=block_days, cache_blocks=cache_blocks)
        val_ds = _CacheDataset(cache_dir, scenarios, val_idx, step=1,
                               block_days=block_days, cache_blocks=cache_blocks)
        logger.info("cache dataset: train %d (step %d), val %d",
                    len(train_ds), step, len(val_ds))
        if len(train_ds) == 0:
            raise RuntimeError(
                "empty training set — the day-index read from the cache "
                "stores found no days in 2015-2100; check the store time "
                "coordinates")
        if len(val_ds) == 0:
            raise RuntimeError("empty validation set — same time-coordinate problem")

        # num_workers=0 + block-local sampler: batches are served from ONE
        # 128-day block each (one zarr chunk read per store per batch),
        # which keeps the loader out of the training loop's way.
        train_dl = DataLoader(train_ds, batch_size=int(tp.get("batch_size", 64)),
                              sampler=_BlockSampler(train_ds.samples, block_days),
                              num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=int(tp.get("batch_size", 64)),
                            shuffle=False, num_workers=0)
        self._train(train_dl, val_dl, tp)

    def predict(self, X, *, target_vars=None, time_chunk=256):
        self.model.eval()
        if target_vars is not None:
            self.target_vars = target_vars
        if self.target_vars is None:
            raise RuntimeError("fit() must be called before predict().")

        if "forcing_scenario" in X.dims:
            if X.sizes["forcing_scenario"] != 1:
                raise ValueError("predict() expects a single scenario.")
            X = X.isel(forcing_scenario=0, drop=True)

        X = X[self.feature_vars]
        device = next(self.model.parameters()).device
        n_time = X.sizes["time"]
        # STREAM to a zarr store chunk-by-chunk and return a LAZY dataset:
        # the in-memory concat held ~11.3 GB of predictions and the starter's
        # reversal then copied it (~25 GB peak) — the kernel was OOM-killed.
        # The lazy path keeps the peak at ~2 GB and the starter's reversal
        # and to_zarr run chunk-wise.
        import tempfile
        from climx_project.cacheio import zarr_fp16_encoding
        stage_enc = zarr_fp16_encoding()
        # Per-process staging dir. Concurrent runs (a notebook training cell
        # executed twice) used to share /tmp/pred_stage.zarr: one run's
        # rmtree + append corrupted the other's store mid-write (observed:
        # the time axis doubled to ~2x31389 days and to_zarr then raised
        # "would overlap multiple Dask chunks"). mkdtemp makes collisions
        # impossible; leftovers die with /tmp at session end.
        stage_path = Path(tempfile.mkdtemp(prefix="pred_stage_")) / "pred.zarr"
        first = True
        n_chunks = (n_time + time_chunk - 1) // time_chunk
        for ci, t0 in enumerate(range(0, n_time, time_chunk)):
            t1 = min(n_time, t0 + time_chunk)
            logger.info("predict chunk %d/%d (days %d-%d)", ci + 1, n_chunks, t0, t1)
            # (feature, time, lat, lon) -> (time, feature, lat, lon): each day
            # is one batch element for the conv net.
            xb = (X.isel(time=slice(t0, t1)).to_array(dim="feature")
                  .transpose("feature", "time", "lat", "lon")
                  .values.astype(np.float32))
            xb = xb.transpose(1, 0, 2, 3)
            # Sub-batch the forward: a 256-day chunk's activations at
            # 192x288 are ~10 GB on the GPU (CUDA OOM observed). 32 days
            # keeps the GPU peak at ~1.2 GB.
            preds = []
            for s0 in range(0, len(xb), 32):
                sub = torch.from_numpy(xb[s0:s0 + 32]).to(device)
                with torch.no_grad():
                    preds.append(self.model(sub).cpu().numpy())
            pred = np.concatenate(preds, axis=0)
            del xb, preds
            tcoord = X["time"].isel(time=slice(t0, t1))
            ds_chunk = xr.Dataset(
                {v: (("time", "lat", "lon"), pred[:, i])
                 for i, v in enumerate(self.target_vars)},
                coords={"time": tcoord, "lat": X["lat"], "lon": X["lon"]},
            ).chunk({"time": 128, "lat": -1, "lon": -1})
            enc = {v: stage_enc for v in ds_chunk.data_vars}
            if first:
                ds_chunk.to_zarr(stage_path, mode="w", encoding=enc)
                first = False
            else:
                ds_chunk.to_zarr(stage_path, mode="a", append_dim="time")
            del pred, ds_chunk
        return xr.open_zarr(stage_path, chunks={"time": 365})

    @classmethod
    def from_checkpoint(cls, ckpt_path, *, feature_vars, target_vars=None):
        """Build an emulator around a Lightning checkpoint (best-val weights).

        Lightning checkpoints don't carry the data-derived feature order —
        pass it (list(X_train.data_vars)) so predict() selects the right
        channels. target_vars defaults to the standard 7-variable order.
        """
        module = UNetModule.load_from_checkpoint(str(ckpt_path))
        em = cls(dict(module.hparams))
        em.model = module
        em.feature_vars = list(feature_vars)
        em.target_vars = list(target_vars or TARGET_VARIABLE_ORDER)
        return em

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_params": self.model_params,
            "model_state": self.model.state_dict(),
            "feature_vars": self.feature_vars,
            "target_vars": self.target_vars,
        }, path)

    def load(self, path):
        # weights_only=False: the dict holds model_params + feature/target
        # var lists, not just tensors (torch>=2.6 defaults weights_only=True).
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_params" in ckpt:
            self.model_params = ckpt["model_params"]
            self.model = UNetModule(**self.model_params)
            self.model.load_state_dict(ckpt["model_state"])
            self.feature_vars = ckpt["feature_vars"]
            self.target_vars = ckpt["target_vars"]
        else:
            # Older/alternate saves: a bare state_dict. Build from the
            # params the emulator was constructed with and load the weights.
            logger.warning("%s has no 'model_params' — treating it as a bare "
                           "state_dict (feature/target vars stay as set)", path)
            self.model.load_state_dict(ckpt)
