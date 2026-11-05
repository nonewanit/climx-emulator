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
                 variable_weights=None, learning_rate=1e-3, pad_to=(16, 32)):
        super().__init__()
        self.save_hyperparameters()
        b = base_channels
        self.pad_h, self.pad_w = int(pad_to[0]), int(pad_to[1])
        self.variable_weights = (
            torch.tensor(variable_weights, dtype=torch.float32)
            if variable_weights is not None else torch.ones(n_out)
        )

        self.enc1 = DoubleConv(n_in, b)
        self.enc2 = DoubleConv(b, 2 * b)
        self.bottleneck = DoubleConv(2 * b, 4 * b)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(4 * b, 2 * b)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
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
        d1 = self.dec1(self.up1(b))
        d2 = self.dec2(self.up2(d1))
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


class UNetEmulator(Emulator):
    def __init__(self, model_params):
        self.model_params = dict(model_params)
        self.model = UNetModule(**self.model_params)
        self.feature_vars = None
        self.target_vars = None

    def fit(self, X_train, y_train, trainer_params=None, **kwargs):
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

        # Local smoke runs: cap the number of days BEFORE materializing the
        # arrays (this machine has 7.5 GB — a full load destabilizes WSL).
        max_days = int(tp.get("max_days", 0) or 0)
        if max_days > 0:
            logger.info("max_days=%d: using a reduced smoke split", max_days)
            X_tr = X_tr.isel(time=slice(0, max_days))
            y_tr = y_tr.isel(time=slice(0, max_days))
            X_val = X_val.isel(time=slice(0, max_days))
            y_val = y_val.isel(time=slice(0, max_days))

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

        trainer_kwargs = {k: v for k, v in tp.items()
                          if k not in ("batch_size", "num_workers",
                                       "model_path", "max_days")}
        trainer_kwargs.setdefault("logger", False)
        model_path = Path(tp.get("model_path", "results/models/unet"))
        model_path.mkdir(parents=True, exist_ok=True)
        trainer_kwargs.setdefault("default_root_dir", str(model_path))

        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks = trainer_kwargs.get("callbacks", [])
        callbacks.append(ModelCheckpoint(
            # val_loss is logged with on_step=False, on_epoch=True -> the key
            # itself holds the epoch aggregate (no "_epoch" suffix for val).
            monitor="val_loss", mode="min", save_top_k=1, save_last=True,
            dirpath=str(model_path), filename="best-{epoch:02d}-{val_loss:.4f}",
        ))
        trainer_kwargs["callbacks"] = callbacks

        self.trainer = pl.Trainer(**trainer_kwargs)
        self.trainer.fit(self.model, train_dl, val_dl)

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
            xb = (X.isel(time=slice(t0, t1)).to_array(dim="feature")
                  .transpose("feature", "time", "lat", "lon")
                  .values.astype(np.float32))
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
