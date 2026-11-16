"""World-model emulator (Phase C): temporal conditioning + autoregressive rollout.

The proposal's Phase C: a tau-day window of past states as additional input —
realized with a ConvLSTM (Shi et al. 2015) at the encoder bottleneck — trained
with teacher forcing, rolled out autoregressively at inference, with a
stability analysis quantifying error accumulation.

Architecture (per day t):
    f_t (22 forcing ch) --[forcing encoder]--> f_feat            (2b, H/4, W/4)
    x_{t-tau..t-1}      --[state encoder, shared]--> tau frames  (2b, H/4, W/4)
    tau frames          --[ConvLSTM, hidden 2b]--> h              (2b, H/4, W/4)
    cat(up(h), up(f_feat)) --[decoder]--> x_t                    (7, H, W)

Training: teacher forcing — the window holds REAL states (x_train data), with
two optional regularizers that make the rollout behave:
  - history_mask_prob: fraction of windows replaced by a learned "missing
    history" prior. This is what inference sees during the spin-up (the first
    tau days of a rollout have no history), so the model must have seen it.
  - scheduled_sampling: fraction of batches whose window is a short rollout of
    the model's OWN predictions (anchored at the true oldest state), so the
    model trains against its own error distribution.

Inference (`predict`): sequential autoregressive rollout. The first tau days
use the learned missing-history prior (zero anomaly = climatology prior, since
the target space is stationarized anomalies); every following day consumes the
previous tau PREDICTED states. Encoded states are cached so each state is
encoded once, not tau times.

Targets live in the starter's SCALED-ANOMALY space, exactly like the U-Net, so
`evaluate()` (reversal, indices, shared tail) works unchanged. Stationarization
reversal is deterministic and additive, so anomaly-space RMSE per day equals
raw-space RMSE per day — `rollout_profile` is directly physical.
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

from .train_utils import run_trainer
from .unet import DEFAULT_VARIABLE_WEIGHTS, TARGET_VARIABLE_ORDER, DoubleConv

logger = logging.getLogger(__name__)

# The 22 preprocessed forcing channels, in the order preprocess.py writes
# X_train.nc (5 GHG + 15 aerosol-split + 2 seasonal). Needed so a checkpoint
# loaded via from_checkpoint() can select X[self.feature_vars]; fit() warns if
# the actual files ever diverge from this contract.
FORCING_VARIABLE_ORDER = [
    "CO2_LBC", "N2O_LBC", "CH4_LBC", "CFC11eq_LBC", "CF2CL2_LBC",
    "BC_AX_positive", "BC_AX_negative", "BC_AX_zeros_mask",
    "BC_N_positive", "BC_N_negative", "BC_N_zeros_mask",
    "SO2_positive", "SO2_negative", "SO2_zeros_mask",
    "SO4_PR_positive", "SO4_PR_negative", "SO4_PR_zeros_mask",
    "OM_NI_positive", "OM_NI_negative", "OM_NI_zeros_mask",
    "sin_doy", "cos_doy",
]


def _pbar(iterable=None, *, desc=None, total=None):
    """Progress bar: tqdm.notebook widgets inside a notebook kernel; plain
    tqdm in a real terminal; DISABLED in notebook subprocesses and redirected
    logs, where \r-updated text bars would render as one line per refresh
    (same capture problem as Lightning's bars — see
    train_utils.is_real_terminal; the callers log progress lines instead)."""
    from tqdm.auto import tqdm
    from .train_utils import in_notebook_kernel, is_real_terminal

    disable = False
    if not in_notebook_kernel():
        disable = not is_real_terminal()
    return tqdm(iterable, desc=desc, total=total, disable=disable)


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell (Shi et al. 2015), no peephole connections."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels,
                              kernel_size, padding=kernel_size // 2)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class WorldModelModule(pl.LightningModule):
    """Temporal-conditioned encoder-decoder (world model) on one day's window."""

    def __init__(self, n_in=22, n_out=7, base_channels=64, tau=7,
                 variable_weights=None, learning_rate=1e-3, pad_to=(16, 32),
                 skip_connections=False, history_mask_prob=0.1,
                 scheduled_sampling=0.0):
        super().__init__()
        self.save_hyperparameters()
        b = base_channels
        self.base = b
        self.pad_h, self.pad_w = int(pad_to[0]), int(pad_to[1])
        self.tau = int(tau)
        self.history_mask_prob = float(history_mask_prob)
        self.scheduled_sampling = float(scheduled_sampling)
        self.skip = bool(skip_connections)
        self.variable_weights = (
            torch.tensor(variable_weights, dtype=torch.float32)
            if variable_weights is not None else torch.ones(n_out)
        )

        # Separate encoders: forcings and states live in different spaces
        # (stationarized forcings vs scaled anomalies). Skip features come
        # from the FORCING encoder only (deterministic driver); the state
        # path conditions purely through the ConvLSTM bottleneck.
        self.encf1 = DoubleConv(n_in, b)
        self.encf2 = DoubleConv(b, 2 * b)
        self.encs1 = DoubleConv(n_out, b)
        self.encs2 = DoubleConv(b, 2 * b)
        self.pool = nn.MaxPool2d(2)

        self.lstm = ConvLSTMCell(2 * b, 2 * b)

        # Learned prior for MISSING history (first tau days of a rollout).
        # Zero-initialized; trained by history_mask_prob windows.
        self.init_state = nn.Parameter(torch.zeros(2 * b, self.pad_h // 4, self.pad_w // 4))

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Decoder input is cat(up(lstm_h), f_feat) = 4b, matching the U-Net's
        # dec1 input width; skip mode concatenates the forcing encoder's
        # features instead (same shapes as the U-Net's skips).
        if self.skip:
            self.dec1 = DoubleConv(4 * b + 2 * b, 2 * b)  # lstm_h + encf2
            self.dec2 = DoubleConv(2 * b + b, b)          # dec1 + encf1
        else:
            self.dec1 = DoubleConv(4 * b, 2 * b)
            self.dec2 = DoubleConv(2 * b, b)
        self.head = nn.Conv2d(b, n_out, 1)

    # ------------------------------------------------------------- helpers

    def _pad_sizes(self, h, w):
        pad_l = (self.pad_w - w) // 2
        pad_r = self.pad_w - w - pad_l
        pad_t = (self.pad_h - h) // 2
        pad_b = self.pad_h - h - pad_t
        return (pad_l, pad_r, pad_t, pad_b)

    def _pad(self, x):
        return nn.functional.pad(x, self._pad_sizes(x.shape[-2], x.shape[-1]))

    def _crop(self, x, h, w):
        return x[:, :, :h, :w]

    def _init_hidden(self, batch, device):
        z = torch.zeros(batch, 2 * self.base, self.pad_h // 4, self.pad_w // 4,
                        device=device)
        return z, torch.zeros_like(z)

    def encode_forcing(self, f_padded):
        """Forcing encoder -> (e1f, e2f, f_feat) on the padded grid."""
        e1f = self.encf1(f_padded)
        e2f = self.encf2(self.pool(e1f))
        return e1f, e2f, self.pool(e2f)

    def encode_state(self, x_padded):
        """State encoder -> bottleneck feature (2b, H/4, W/4)."""
        return self.pool(self.encs2(self.pool(self.encs1(x_padded))))

    def decode(self, h, e1f, e2f, f_feat):
        """Bottleneck hidden state + forcing features -> padded full-res fields.

        h and f_feat both live at (2b, H/4, W/4); both are upsampled before
        the concat so dec1 operates at H/2 and dec2 at H — the U-Net's exact
        decoder layout (dec1 input width 4b).
        """
        if self.skip:
            d1 = self.dec1(torch.cat([self.up1(h), e2f], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d1), e1f], dim=1))
        else:
            d1 = self.dec1(torch.cat([self.up1(h), self.up1(f_feat)], dim=1))
            d2 = self.dec2(self.up2(d1))
        return self.head(d2)

    # ------------------------------------------------------------- forward

    def forward(self, f, history, mask=None):
        """One day: f (B, n_in, H, W), history (B, tau, n_out, H, W).

        `mask` (B, tau) bool marks history slots replaced by the learned
        missing-history prior (True = replace).
        """
        batch = f.shape[0]
        h_in, w_in = f.shape[-2:]
        f = self._pad(f)
        history = self._pad(history)
        e1f, e2f, f_feat = self.encode_forcing(f)

        if mask is None:
            mask = torch.zeros(batch, self.tau, dtype=torch.bool, device=f.device)
        h, c = self._init_hidden(batch, f.device)
        for k in range(self.tau):
            feat = self.encode_state(history[:, k])
            feat = torch.where(mask[:, k].view(batch, 1, 1, 1),
                               self.init_state.unsqueeze(0), feat)
            h, (h, c) = self.lstm(feat, (h, c))
        return self._crop(self.decode(h, e1f, e2f, f_feat), h_in, w_in)

    def _rollout_history(self, f, x0):
        """tau-step rollout of the model's OWN predictions, anchored at the true
        oldest state x0. Forcing features of day t are reused for every step
        (the batch carries no per-day forcing window; forcings vary slowly).

        Gradients flow through the rollout — this is scheduled sampling's
        honest form: the model trains against its own error distribution.
        """
        batch = f.shape[0]
        h_in, w_in = f.shape[-2:]
        f = self._pad(f)
        e1f, e2f, f_feat = self.encode_forcing(f)
        h, c = self._init_hidden(batch, f.device)
        x = self._pad(x0)
        states = []
        for _ in range(self.tau):
            feat = self.encode_state(x)
            h, (h, c) = self.lstm(feat, (h, c))
            x = self.decode(h, e1f, e2f, f_feat)
            states.append(self._crop(x, h_in, w_in))
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def rollout_step(self, f, state_feats):
        """One inference day: f (1, n_in, H, W) unpadded; state_feats is a list
        of tau ENCODED state features (1, 2b, H/4, W/4), None for missing days
        (-> learned prior). Returns (pred (1, n_out, H, W), pred_feat)."""
        h_in, w_in = f.shape[-2:]
        f = self._pad(f)
        e1f, e2f, f_feat = self.encode_forcing(f)
        h, c = self._init_hidden(1, f.device)
        for feat in state_feats:
            if feat is None:
                feat = self.init_state.unsqueeze(0)
            h, (h, c) = self.lstm(feat, (h, c))
        out_p = self.decode(h, e1f, e2f, f_feat)
        pred_feat = self.encode_state(out_p)
        return self._crop(out_p, h_in, w_in), pred_feat

    # ------------------------------------------------------------- Lightning

    def _step(self, batch, split, mask=None):
        f, history, y = batch
        y_hat = self(f, history, mask=mask)
        weights = self.variable_weights.to(y_hat.device)
        loss = (weights.view(1, -1, 1, 1) * (y_hat - y) ** 2).mean()
        # sync_dist=True: under DDP each rank otherwise logs only its own
        # subset of the val batches, biasing early stopping and the best-ckpt
        # choice (no-op on a single device).
        self.log(f"{split}_loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        f, history, y = batch
        mask = None
        if self.history_mask_prob > 0:
            full = torch.rand(f.shape[0], device=f.device) < self.history_mask_prob
            mask = full.unsqueeze(1).expand(-1, self.tau)
        if self.scheduled_sampling > 0 and \
                torch.rand(1, device=f.device).item() < self.scheduled_sampling:
            history = self._rollout_history(f, history[:, 0])
            mask = None
        return self._step((f, history, y), "train", mask=mask)

    def validation_step(self, batch, batch_idx):
        # Teacher-forced only: val_loss stays comparable with the U-Net's.
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Same LR policy as UNetModule (see 564a050): constant-LR training
        # plateaus well above what the model can reach; halve the LR when
        # val_loss stalls for 4 epochs.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss",
                                 "interval": "epoch",
                                 "frequency": 1}}


class _WindowDataset(Dataset):
    """(f_t, history, y_t) windows over per-scenario in-memory arrays.

    xs[scen]: (T, F, H, W), ys[scen]: (T, 7, H, W) — same stationarized
    (scaled-anomaly) space. History is a VIEW into the per-scenario array
    (leading-dim slice, contiguous), so no tau-fold memory blow-up.
    """

    def __init__(self, xs: dict, ys: dict, tau: int):
        self.tau = int(tau)
        self.xs, self.ys = xs, ys
        self.samples = []
        for scen, xa in xs.items():
            t_len = xa.shape[0]
            assert ys[scen].shape[0] == t_len, f"{scen}: x/y day mismatch"
            self.samples.extend((scen, t) for t in range(self.tau, t_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        scen, t = self.samples[i]
        x = self.xs[scen]
        y = self.ys[scen]
        return (torch.from_numpy(x[t]),
                torch.from_numpy(y[t - self.tau:t]),
                torch.from_numpy(y[t]))


class WorldModelEmulator(Emulator):
    def __init__(self, model_params):
        self.model_params = dict(model_params)
        self.model = WorldModelModule(**self.model_params)
        self.feature_vars = None
        self.target_vars = None

    # ------------------------------------------------------------- training

    def fit(self, X_train, y_train, trainer_params=None, **kwargs):
        # The zarr-cache path (full-res) is not ported yet; windows need a
        # windowed cache dataset. Lite-first per the proposal's staging.
        if isinstance(X_train, (str, Path)):
            raise NotImplementedError(
                "worldmodel.fit from a cache dir is not implemented yet; "
                "pass the preprocessed lite datasets (the windowed cache "
                "dataset comes with the full-res port).")
        self.feature_vars = list(X_train.data_vars)
        self.target_vars = list(y_train.data_vars)
        if self.feature_vars != FORCING_VARIABLE_ORDER:
            logger.warning(
                "X_train data_vars order differs from FORCING_VARIABLE_ORDER — "
                "checkpoint-loaded predict() would select the WRONG channels. "
                "Regenerate preprocessing or update the constant in "
                "models/worldmodel.py.")

        tp = dict(trainer_params or {})
        tau = int(self.model_params.get("tau", 7))
        max_days = int(tp.get("max_days", 0) or 0)

        # Same split convention as the starter's NN / the U-Net (projections
        # only, all scenarios); val starts at 2081 to avoid the starter's
        # overlapping 2080 year.
        if max_days > 0:
            n = int(X_train.sizes["time"])
            cut = max(tau + 2, int(n * 0.8))
            val_n = max(tau + 2, max_days // 4)
            logger.info("smoke split: %d days total, train %d / val %d "
                        "(max_days=%d, tau=%d)", n, cut, val_n, max_days, tau)
            X_tr_ds = X_train.isel(time=slice(0, min(cut, max_days)))
            y_tr_ds = y_train.isel(time=slice(0, min(cut, max_days)))
            X_val_ds = X_train.isel(time=slice(cut, cut + val_n))
            y_val_ds = y_train.isel(time=slice(cut, cut + val_n))
        else:
            # Val arrays start tau days BEFORE 2081 so every val window has
            # real history (no learned-prior confound in teacher-forced eval).
            import pandas as pd
            val_start = str((pd.Timestamp("2081-01-01")
                             - pd.Timedelta(days=tau)).date())
            X_tr_ds = X_train.sel(time=slice("2015-01-01", "2080-12-31"))
            y_tr_ds = y_train.sel(time=slice("2015-01-01", "2080-12-31"))
            X_val_ds = X_train.sel(time=slice(val_start, "2100-12-31"))
            y_val_ds = y_train.sel(time=slice(val_start, "2100-12-31"))

        logger.info("Loading world-model window arrays into memory ...")
        xs_tr = _to_numpy_per_scenario(X_tr_ds)
        ys_tr = _to_numpy_per_scenario(y_tr_ds)
        xs_val = _to_numpy_per_scenario(X_val_ds)
        ys_val = _to_numpy_per_scenario(y_val_ds)
        for scen in xs_tr:
            logger.info("%s: train days %d, val days %d",
                        scen, xs_tr[scen].shape[0], xs_val[scen].shape[0])

        train_ds = _WindowDataset(xs_tr, ys_tr, tau)
        val_ds = _WindowDataset(xs_val, ys_val, tau)
        logger.info("train windows %d, val windows %d (tau=%d)",
                    len(train_ds), len(val_ds), tau)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise RuntimeError("empty window dataset — need >= tau+1 days "
                               "per scenario (raise max_days or lower tau)")

        train_dl = DataLoader(train_ds, batch_size=int(tp.get("batch_size", 64)),
                              shuffle=True,
                              num_workers=int(tp.get("num_workers", 0)))
        val_dl = DataLoader(val_ds, batch_size=int(tp.get("batch_size", 64)),
                            shuffle=False,
                            num_workers=int(tp.get("num_workers", 0)))

        self.model = run_trainer(
            self.model,
            lambda ckpt: WorldModelModule.load_from_checkpoint(ckpt),
            train_dl, val_dl, tp)

    # ------------------------------------------------------------- inference

    def predict(self, X, *, target_vars=None):
        """Autoregressive rollout over the full test period.

        Sequential: each day consumes the previous tau PREDICTED states; the
        first tau days use the learned missing-history prior (zero anomaly =
        climatology prior in this stationarized space). Encoded states are
        cached so each state is encoded once, not tau times. Returns an
        in-memory Dataset (lite-sized); the full-res port must stream to
        zarr like UNetEmulator.predict.
        """
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
        arr = (X.to_array(dim="feature")
               .transpose("feature", "time", "lat", "lon")
               .values.astype(np.float32))
        arr = np.ascontiguousarray(arr.transpose(1, 0, 2, 3))  # (T, F, H, W)
        n_time = arr.shape[0]
        tau = self.model.tau
        h_grid, w_grid = arr.shape[2], arr.shape[3]

        out = np.empty((n_time, len(self.target_vars), h_grid, w_grid),
                       dtype=np.float32)
        feat_cache: dict[int, torch.Tensor] = {}
        bar = _pbar(range(n_time), desc="rollout predict")
        with torch.no_grad():
            for t in bar:
                if bar.disable and t % 100 == 0:
                    logger.info("rollout day %d/%d", t, n_time)
                f = torch.from_numpy(arr[t]).unsqueeze(0).to(device)
                feats = [feat_cache.get(t - tau + k) for k in range(tau)]
                pred, pred_feat = self.model.rollout_step(f, feats)
                feat_cache[t] = pred_feat
                feat_cache.pop(t - tau, None)
                out[t] = pred[0].cpu().numpy()

        return xr.Dataset(
            {v: (("time", "lat", "lon"), out[:, i])
             for i, v in enumerate(self.target_vars)},
            coords={"time": X["time"], "lat": X["lat"], "lon": X["lon"]},
        )

    def rollout_profile(self, X, y, horizon_days=None, batch_days=256):
        """Error-accumulation profile (the Phase C stability analysis).

        X: (T, F, H, W) numpy forcings; y: (T, 7, H, W) numpy targets, in the
        stationarized (scaled-anomaly) space. Returns
            {"days": 0..n-1, "one_step_rmse": [...], "rollout_rmse": [...]}
        per-day RMSE over all 7 variables, UNWEIGHTED. one_step_rmse is NaN
        for the first tau days (teacher forcing has no history there); the
        rollout starts from the missing-history prior, so its early days
        quantify the spin-up transient and the rest the long-horizon drift.
        Anomaly-space RMSE == raw-space daily RMSE (the reversal adds the
        same daily climatology to both sides).
        """
        tau = self.model.tau
        device = next(self.model.parameters()).device
        self.model.eval()
        n = len(y) if horizon_days is None else min(int(horizon_days), len(y))
        if X.shape[0] < n or y.shape[0] < n:
            raise ValueError(f"need >= {n} days in X ({X.shape[0]}) and y ({y.shape[0]})")

        one_step = np.full(n, np.nan, dtype=np.float32)
        with torch.no_grad():
            for t0 in _pbar(range(tau, n, batch_days), desc="teacher-forced"):
                t1 = min(n, t0 + batch_days)
                f = torch.from_numpy(X[t0:t1]).to(device)
                hist = torch.from_numpy(np.stack(
                    [y[t0 - tau + k: t1 - tau + k] for k in range(tau)],
                    axis=1)).to(device)
                y_true = torch.from_numpy(y[t0:t1]).to(device)
                err = (self.model(f, hist) - y_true).square().mean(dim=(1, 2, 3)).sqrt()
                one_step[t0:t1] = err.cpu().numpy()

            rollout = np.empty(n, dtype=np.float32)
            feat_cache: dict[int, torch.Tensor] = {}
            bar = _pbar(range(n), desc="autoregressive")
            for t in bar:
                if bar.disable and t % 100 == 0:
                    logger.info("profile rollout day %d/%d", t, n)
                f = torch.from_numpy(X[t]).unsqueeze(0).to(device)
                feats = [feat_cache.get(t - tau + k) for k in range(tau)]
                pred, pred_feat = self.model.rollout_step(f, feats)
                feat_cache[t] = pred_feat
                feat_cache.pop(t - tau, None)
                y_true = torch.from_numpy(y[t]).unsqueeze(0).to(device)
                rollout[t] = (pred - y_true).square().mean().sqrt().item()

        return {"days": np.arange(n), "one_step_rmse": one_step,
                "rollout_rmse": rollout}

    @classmethod
    def from_checkpoint(cls, ckpt_path):
        """Build an emulator around a Lightning checkpoint (best-val weights).

        Lightning checkpoints don't carry the data-derived feature order, so
        feature_vars falls back to FORCING_VARIABLE_ORDER (the preprocessing
        contract, cross-checked by fit()).
        """
        module = WorldModelModule.load_from_checkpoint(str(ckpt_path))
        em = cls(dict(module.hparams))
        em.model = module
        em.feature_vars = list(FORCING_VARIABLE_ORDER)
        em.target_vars = list(TARGET_VARIABLE_ORDER)
        return em

    # ------------------------------------------------------------- persistence

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
            self.model = WorldModelModule(**self.model_params)
            self.model.load_state_dict(ckpt["model_state"])
            self.feature_vars = ckpt["feature_vars"]
            self.target_vars = ckpt["target_vars"]
        else:
            # Older/alternate saves: a bare state_dict. Build from the
            # params the emulator was constructed with and load the weights.
            logger.warning("%s has no 'model_params' — treating it as a bare "
                           "state_dict (feature/target vars stay as set)", path)
            self.model.load_state_dict(ckpt)


def _to_numpy_per_scenario(ds: xr.Dataset) -> dict:
    """xarray (forcing_scenario, time, lat, lon) -> {scen: (T, F, H, W) fp32}."""
    arr = (ds.to_array(dim="feature")
           .transpose("forcing_scenario", "time", "feature", "lat", "lon")
           .values.astype(np.float32))
    names = [str(s) for s in ds.forcing_scenario.values]
    return {names[i]: np.ascontiguousarray(arr[i]) for i in range(len(names))}
