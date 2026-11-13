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
                 skip_connections=True):
        super().__init__()
        self.save_hyperparameters()
        b = base_channels
        self.pad_h, self.pad_w = int(pad_to[0]), int(pad_to[1])
        self.skip = bool(skip_connections)
        self.variable_weights = (
            torch.tensor(variable_weights, dtype=torch.float32)
            if variable_weights is not None else torch.ones(n_out)
        )

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
        weights = self.variable_weights.to(y_hat.device)
        loss = (weights.view(1, -1, 1, 1) * (y_hat - y) ** 2).mean()
        self.log(f"{split}_loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


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


class _CacheDataset(Dataset):
    """Map-style dataset over the fp16 scenario-pure cache (scripts/full_cache.py).

    Serves (scenario, day) samples without materializing anything: each zarr
    time-block is loaded once into a small LRU buffer and its days are served
    from memory. `step` subsamples days at LOAD time (daily physics kept —
    never averaged), so the same ×1 cache feeds Phase B (step=3) and Phase C
    (step=1, contiguous).
    """

    def __init__(self, cache_dir: Path, scenarios, day_indices, step: int = 1,
                 block_days: int = 64, cache_blocks: int = 6):
        self.block_days = block_days
        self.cache: dict = {}
        self.lru: list = []
        self.cache_blocks = cache_blocks
        self.x_paths = {}
        self.y_paths = {}
        self.time = {}
        for scen in scenarios:
            self.x_paths[scen] = sorted((cache_dir / f"X_{scen}").glob("*.zarr")) \
                if (cache_dir / f"X_{scen}").is_dir() \
                else sorted(cache_dir.glob(f"X_*_{scen}.zarr"))
            self.y_paths[scen] = sorted(cache_dir.glob(f"Y_*_{scen}.zarr")) or \
                sorted(cache_dir.glob(f"y_*_{scen}.zarr"))
            if not self.x_paths[scen] or not self.y_paths[scen]:
                raise FileNotFoundError(
                    f"No cache files for scenario {scen} under {cache_dir}")
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
            # each store holds one variable: the array named in the file stem
            ds = xr.open_zarr(p).isel(time=tsel)
            name = list(ds.data_vars)[0]
            arr = ds[name].values.astype(np.float32)
            if arr.shape[0] == 0:
                raise RuntimeError(
                    f"{p.name}: empty read at time {tsel}; opened sizes "
                    f"{dict(ds.sizes)}")
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
        """Shared trainer plumbing: callback setup, fit, best-checkpoint load."""
        trainer_kwargs = {k: v for k, v in tp.items()
                          if k not in ("batch_size", "num_workers",
                                       "model_path", "max_days",
                                       "early_stop_patience",
                                       "scenarios", "sample_step",
                                       "block_days", "cache_blocks")}
        trainer_kwargs.setdefault("logger", False)
        model_path = Path(tp.get("model_path", "results/models/unet"))
        model_path.mkdir(parents=True, exist_ok=True)
        trainer_kwargs.setdefault("default_root_dir", str(model_path))

        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        callbacks = trainer_kwargs.get("callbacks", [])
        callbacks.append(ModelCheckpoint(
            # val_loss is logged with on_step=False, on_epoch=True -> the key
            # itself holds the epoch aggregate (no "_epoch" suffix for val).
            monitor="val_loss", mode="min", save_top_k=1, save_last=True,
            dirpath=str(model_path), filename="best-{epoch:02d}-{val_loss:.4f}",
        ))
        # Self-regulating long runs: stop when validation loss plateaus, so
        # max_epochs can be set generously without wasting GPU time.
        callbacks.append(EarlyStopping(
            monitor="val_loss", mode="min",
            patience=int(tp.get("early_stop_patience", 3)), min_delta=1e-4,
            verbose=True,
        ))
        trainer_kwargs["callbacks"] = callbacks

        self.trainer = pl.Trainer(**trainer_kwargs)
        self.trainer.fit(self.model, train_dl, val_dl)

        # Evaluate with the BEST-val weights, not the last epoch's — val loss
        # fluctuates at this scale and last != best.
        best = getattr(self.trainer.checkpoint_callback, "best_model_path", None)
        if best:
            self.model = UNetModule.load_from_checkpoint(best)
            logger.info("Loaded best-val checkpoint for evaluation: %s", best)

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

        # num_workers=0: zarr v3 runs an internal asyncio loop; forked
        # DataLoader workers from a process that already opened zarr stores
        # produced empty reads (observed). The block reads are local-disk
        # and cheap, so single-process loading is fine.
        train_dl = DataLoader(train_ds, batch_size=int(tp.get("batch_size", 64)),
                              shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=int(tp.get("batch_size", 64)),
                            shuffle=False, num_workers=0)
        self._train(train_dl, val_dl, tp)

    def predict(self, X, *, target_vars=None, time_chunk=512):
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
        chunks = []
        n_time = X.sizes["time"]
        for t0 in range(0, n_time, time_chunk):
            t1 = min(n_time, t0 + time_chunk)
            # (feature, time, lat, lon) -> (time, feature, lat, lon): each day
            # is one batch element for the conv net.
            xb = (X.isel(time=slice(t0, t1)).to_array(dim="feature")
                  .transpose("feature", "time", "lat", "lon")
                  .values.astype(np.float32))
            xb = xb.transpose(1, 0, 2, 3)
            with torch.no_grad():
                pred = self.model(torch.from_numpy(xb).to(device)).cpu().numpy()
            tcoord = X["time"].isel(time=slice(t0, t1))
            chunks.append(xr.Dataset(
                {v: (("time", "lat", "lon"), pred[:, i])
                 for i, v in enumerate(self.target_vars)},
                coords={"time": tcoord, "lat": X["lat"], "lon": X["lon"]},
            ))
        return xr.concat(chunks, dim="time")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_params": self.model_params,
            "model_state": self.model.state_dict(),
            "feature_vars": self.feature_vars,
            "target_vars": self.target_vars,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.model_params = ckpt["model_params"]
        self.model = UNetModule(**self.model_params)
        self.model.load_state_dict(ckpt["model_state"])
        self.feature_vars = ckpt["feature_vars"]
        self.target_vars = ckpt["target_vars"]
