"""Shared code for the Session 2 RNN notebooks (marimo RNN.py, RNN_colab.ipynb, tools/pretrain_lstm.py).

Hourly weather for Chachapoyas (Amazonas, Peru) from NASA POWER, forecasting windows, naive baselines and a
small recurrent network (LSTM, GRU or plain RNN).
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_RAW = "https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/session2_inside_the_black_box/src"
WEATHER_FILE = "data/chachapoyas_hourly.csv"
WEIGHTS_FILE = "data/lstm_chachapoyas.pt"
CHACHAPOYAS = (-6.23, -77.87)  # latitude, longitude

# Column in the CSV -> (NASA POWER parameter, label for plots, unit)
VARIABLES = {
    "temperature": ("T2M", "Temperature", "°C"),
    "humidity": ("RH2M", "Relative humidity", "%"),
    "rain": ("PRECTOTCORR", "Rainfall", "mm/h"),
    "wind": ("WS2M", "Wind speed", "m/s"),
    "pressure": ("PS", "Surface pressure", "kPa"),
}
HOUR_FEATURES = ["hour_sin", "hour_cos"]  # time of day as a smooth cycle (23:00 is close to 00:00)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch(rel_path, base_dir=None, cache_dir=None):
    """Return a local path to `rel_path`: next to this file if present, else downloaded (and cached)."""
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(base_dir, rel_path)
    if os.path.exists(local):
        return local
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "untrm_workshop")
    cached = os.path.join(cache_dir, os.path.basename(rel_path))
    if not os.path.exists(cached):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(f"{REPO_RAW}/{rel_path}", cached)
    return cached


def fetch_nasa_power(lat, lon, start="20230101", end="20241231"):
    """Download hourly weather for one location from NASA POWER (no account needed).

    Returns a DataFrame with a `time` column (local solar time) and one column per entry of VARIABLES.
    """
    params = ",".join(v[0] for v in VARIABLES.values())
    url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters={params}&community=AG"
           f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON&time-standard=LST")
    with urllib.request.urlopen(url, timeout=300) as response:
        series = json.load(response)["properties"]["parameter"]
    first = series[VARIABLES["temperature"][0]]
    df = pd.DataFrame({"time": pd.to_datetime(list(first), format="%Y%m%d%H")})
    for name, (code, _, _) in VARIABLES.items():
        df[name] = np.array(list(series[code].values()), dtype=np.float32)
    df = df.replace(-999.0, np.nan).interpolate(limit_direction="both")  # -999 marks missing values
    return df


def load_weather(path):
    """Read the hourly CSV and add the time-of-day features."""
    df = pd.read_csv(path, parse_dates=["time"])
    hour = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    return df


def split_bounds(n, fractions=(0.7, 0.15, 0.15)):
    """Chronological split: first 70% train, next 15% validation, last 15% test (indices into the series)."""
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    return {"train": (0, a), "validation": (a, b), "test": (b, n)}


class Forecaster:
    """Turns an hourly table into (past H hours of inputs) -> (target F hours after the last observed hour)."""

    def __init__(self, df, target="temperature", inputs=("temperature", "humidity", "rain"), history=48,
                 horizon=6, use_hour=True):
        self.df, self.target, self.history, self.horizon = df, target, history, horizon
        self.inputs = list(inputs) + (HOUR_FEATURES if use_hour else [])
        self.bounds = split_bounds(len(df))
        train = df.iloc[slice(*self.bounds["train"])]
        self.mean = train[self.inputs].mean().values.astype(np.float32)  # normalize with training data only
        self.std = train[self.inputs].std().values.astype(np.float32) + 1e-6
        self.t_mean = float(train[target].mean())
        self.t_std = float(train[target].std())

    def windows(self, split, stride=1):
        """Return dict with X (N, H, n_inputs) normalized, y (N,) in physical units, and baselines."""
        lo, hi = self.bounds[split]
        x = ((self.df[self.inputs].values[lo:hi] - self.mean) / self.std).astype(np.float32)
        t = self.df[self.target].values[lo:hi].astype(np.float32)
        H, F = self.history, self.horizon
        now = np.arange(H - 1, len(t) - F, stride)  # index of the last observed hour
        X = np.stack([x[i - H + 1:i + 1] for i in now])
        return {
            "X": torch.from_numpy(X),
            "y": t[now + F],
            "persistence": t[now],               # "it will be the same as now"
            "yesterday": t[now + F - 24],        # "same hour yesterday" (needs F <= 24)
            "start": lo + now - H + 1,           # row of the first hour of each window
            "now": lo + now,
        }

    def normalize_target(self, y):
        return (y - self.t_mean) / self.t_std

    def denormalize_target(self, y):
        return y * self.t_std + self.t_mean


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WeatherRNN(nn.Module):
    """A recurrent layer (LSTM, GRU or plain RNN) reads the window hour by hour; a linear layer makes the forecast."""

    CELLS = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}

    def __init__(self, n_inputs, hidden=32, cell="LSTM"):
        super().__init__()
        self.cell = cell
        self.rnn = self.CELLS[cell](n_inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def hidden_states(self, x):
        """Hidden state after every hour: (batch, hours, hidden)."""
        out, _ = self.rnn(x)
        return out

    def forward(self, x):
        return self.head(self.hidden_states(x)[:, -1]).squeeze(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def predict(model, X, forecaster, batch_size=1024):
    """Forecast in physical units for windows X."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(X[i:i + batch_size]))
    return forecaster.denormalize_target(torch.cat(out).numpy())


def train_model(model, forecaster, train, val, epochs=10, lr=3e-3, batch_size=128, seed=0, log=None, device="cpu"):
    """Train with Adam on MSE (normalized target). History holds per-epoch train/validation MAE in physical units."""
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_train = torch.from_numpy(forecaster.normalize_target(train["y"]).astype(np.float32))
    history = {"train_mae": [], "val_mae": []}
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y_train))
        abs_err = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb, yb = train["X"][idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = nn.functional.mse_loss(out, yb)
            loss.backward()
            optimizer.step()
            abs_err += (out - yb).abs().sum().item()
        model.cpu()
        history["train_mae"].append(abs_err / len(order) * forecaster.t_std)
        history["val_mae"].append(mae(predict(model, val["X"], forecaster), val["y"]))
        model.to(device)
        if log:
            log(f"Epoch {epoch + 1}/{epochs}: train MAE {history['train_mae'][-1]:.2f}, "
                f"validation MAE {history['val_mae'][-1]:.2f} ({time.time() - start:.0f} s)")
    history["seconds"] = time.time() - start
    model.cpu()
    return history
