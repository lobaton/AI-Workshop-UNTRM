"""Shared code for the Session 2 autoencoder notebooks (marimo Autoencoder.py, Autoencoder_colab.ipynb,
tools/pretrain_ae.py).

Uses the Chachapoyas hourly weather from rnn_common as a "field sensor": one input = one day (24 hours) of
temperature and humidity. The autoencoder learns normal days; faults are injected into test days.
"""
import time

import numpy as np
import torch
from torch import nn

import rnn_common as rc

WEIGHTS_FILE = "data/ae_chachapoyas.pt"
COLUMNS = ["temperature", "humidity"]
FAULTS = {
    "Stuck sensor": "the reading freezes at one value for several hours",
    "Spike": "a single wrong reading, much too high",
    "Drift": "the reading slowly creeps away from the true value",
    "Dropout to zero": "the sensor reports 0 for a few hours",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class SensorDays:
    """Day-long windows (channels x 24 hours) scaled with the training period's mean and spread."""

    def __init__(self, df, columns=COLUMNS):
        self.df, self.columns = df, list(columns)
        self.bounds = rc.split_bounds(len(df))
        train = df[self.columns].values[slice(*self.bounds["train"])]
        self.mean = train.mean(axis=0).astype(np.float32)
        self.std = train.std(axis=0).astype(np.float32) + 1e-6

    def scale(self, values):
        """(..., 24, channels) in physical units -> (..., channels, 24) scaled."""
        return np.swapaxes((values - self.mean) / self.std, -1, -2).astype(np.float32)

    def unscale(self, x):
        """Inverse of scale: (..., channels, 24) -> (..., 24, channels) in physical units."""
        return np.swapaxes(x, -1, -2) * self.std + self.mean

    def sliding(self, split, stride=1):
        """All 24-hour windows of a split, sliding by `stride` hours (used for training)."""
        lo, hi = self.bounds[split]
        values = self.df[self.columns].values[lo:hi]
        starts = np.arange(0, len(values) - 24 + 1, stride)
        return np.stack([values[s:s + 24] for s in starts])

    def calendar_days(self, split):
        """Whole calendar days (00:00-23:00) of a split, plus each day's start row."""
        lo, hi = self.bounds[split]
        hours = self.df["time"].dt.hour.values
        starts = [s for s in range(lo, hi - 23) if hours[s] == 0]
        values = self.df[self.columns].values
        return np.stack([values[s:s + 24] for s in starts]), np.array(starts)


def inject_fault(day, kind, size, rng, channel=0):
    """Return a copy of `day` (24 x channels, physical units) with one fault in `channel`.

    `size` goes from 1 (subtle) to 5 (obvious).
    """
    day = day.copy()
    x = day[:, channel]
    if kind == "Stuck sensor":
        length = 3 + 3 * size
        s = int(rng.integers(0, 24 - length + 1))
        x[s:s + length] = x[s]
    elif kind == "Spike":
        s = int(rng.integers(1, 23))
        x[s] += 2.0 * size + 2.0
    elif kind == "Drift":
        x += np.linspace(0, 1.2 * size + 0.5, 24)
    elif kind == "Dropout to zero":
        length = 1 + size
        s = int(rng.integers(0, 24 - length + 1))
        x[s:s + length] = 0.0
    else:
        raise ValueError(kind)
    return day


def faulty_test_set(days, kind, size, seed=0, fraction=0.5):
    """Inject `kind` faults into a random `fraction` of the days. Returns (days, labels 1 = faulty)."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(len(days), dtype=int)
    labels[rng.permutation(len(days))[: int(round(fraction * len(days)))]] = 1
    out = np.stack([inject_fault(d, kind, size, rng) if l else d for d, l in zip(days, labels)])
    return out, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SensorAE(nn.Module):
    """1D convolutional autoencoder: (channels, 24 hours) -> latent numbers -> (channels, 24 hours)."""

    def __init__(self, latent=4, channels=2):
        super().__init__()
        self.latent = latent
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),  # 24 -> 12 hours
            nn.Flatten(), nn.Linear(32 * 12, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32 * 12), nn.ReLU(), nn.Unflatten(1, (32, 12)),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),  # 12 -> 24 hours
            nn.Conv1d(16, channels, 5, padding=2),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


LAYER_NAMES = {
    "Conv1d": lambda l: f"Conv1d ({l.out_channels} filters, width {l.kernel_size[0]}"
                        + (f", stride {l.stride[0]})" if l.stride[0] > 1 else ")"),
    "ConvTranspose1d": lambda l: f"ConvTranspose1d ({l.out_channels} filters, width {l.kernel_size[0]}, "
                                 f"stride {l.stride[0]})",
    "Linear": lambda l: f"Linear ({l.in_features} → {l.out_features})",
    "Flatten": lambda l: "Flatten",
    "Unflatten": lambda l: "Unflatten",
}


def layer_table(model):
    """Rows of (part, layer, output shape, parameters); a ReLU is merged into the layer before it."""
    rows = []
    x = torch.zeros(1, model.encoder[0].in_channels, 24)
    with torch.no_grad():
        for part, seq in (("Encoder", model.encoder), ("Decoder", model.decoder)):
            for layer in seq:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    rows[-1] = (rows[-1][0], rows[-1][1] + " + ReLU", rows[-1][2], rows[-1][3])
                    continue
                name = LAYER_NAMES[type(layer).__name__](layer)
                rows.append((part, name, tuple(x.shape[1:]), sum(p.numel() for p in layer.parameters())))
    return rows


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training and scoring
# ---------------------------------------------------------------------------
def reconstruct(model, x, batch_size=2048):
    """Reconstructions of scaled windows x (N, channels, 24)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out.append(model(torch.as_tensor(x[i:i + batch_size])).numpy())
    return np.concatenate(out)


def reconstruction_error(model, x):
    """Mean squared reconstruction error per window (scaled units)."""
    return ((reconstruct(model, x) - x) ** 2).mean(axis=(1, 2))


def train_ae(model, x_train, x_val, epochs=20, lr=3e-3, batch_size=128, seed=0, log=None):
    """Train to reconstruct the input (MSE). History holds per-epoch training and validation loss."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_train = torch.as_tensor(x_train)
    history = {"train_loss": [], "val_loss": []}
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(order), batch_size):
            xb = x_train[order[i:i + batch_size]]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history["train_loss"].append(total / len(order))
        history["val_loss"].append(float(reconstruction_error(model, x_val).mean()))
        if log:
            log(f"Epoch {epoch + 1}/{epochs}: train loss {history['train_loss'][-1]:.4f}, "
                f"validation loss {history['val_loss'][-1]:.4f} ({time.time() - start:.0f} s)")
    history["seconds"] = time.time() - start
    return history
