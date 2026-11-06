"""Shared code for the Session 2 CNN notebooks (marimo CNN.py, CNN_colab.ipynb, tools/pretrain_cnn.py).

Keeping the model, data loading and training loop in one place means the network the presenter shows
is exactly the network participants train.
"""
import io
import os
import time
import urllib.request

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO_RAW = "https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/session2_inside_the_black_box/src"
BEANS_FILE = "data/beans128.npz"
WEIGHTS_FILE = "data/beans_cnn128.pt"

CLASS_NAMES = ["angular leaf spot", "bean rust", "healthy"]
SPLITS = {"train": 0, "validation": 1, "test": 2}


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


def load_beans(path):
    """Load the 128x128 beans pack. Returns dict of split -> (images uint8 (N,H,W,3), labels int64)."""
    pack = np.load(path)
    jpeg, offsets = pack["jpeg"], pack["offsets"]
    images = np.stack([
        np.asarray(Image.open(io.BytesIO(jpeg[offsets[i]:offsets[i + 1]].tobytes())).convert("RGB"))
        for i in range(len(offsets) - 1)
    ])
    labels, split = pack["labels"], pack["split"]
    return {name: (images[split == k], labels[split == k]) for name, k in SPLITS.items()}


def to_tensor(images):
    """uint8 (N,H,W,3) -> float tensor (N,3,H,W) scaled to [-1, 1] (as in the CIFAR10 script)."""
    x = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


def to_image(x):
    """Inverse of to_tensor for one image: (3,H,W) tensor -> (H,W,3) float array in [0, 1]."""
    return (x.detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)


def random_subset(images, labels, n, seed=0):
    """Pick n examples, keeping the classes balanced."""
    if n >= len(labels):
        return images, labels
    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.permutation(np.flatnonzero(labels == c))[: n // len(CLASS_NAMES)]
        for c in range(len(CLASS_NAMES))
    ])
    return images[idx], labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BeansCNN(nn.Module):
    """Small CNN: three conv blocks (conv -> ReLU -> max pool), global average pool, linear classifier."""

    def __init__(self, n_filters=16, n_classes=3):
        super().__init__()
        f = n_filters
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, f, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2)),
            nn.Sequential(nn.Conv2d(f, 2 * f, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2)),
            nn.Sequential(nn.Conv2d(2 * f, 4 * f, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4 * f, n_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.pool(x).flatten(1))

    def feature_maps(self, x):
        """Activations after the ReLU of each conv block (before pooling)."""
        maps = []
        for block in self.blocks:
            x = block[1](block[0](x))
            maps.append(x)
            x = block[2](x)
        return maps


def layer_table(model, image_size=128):
    """Rows of (layer, output shape, parameters), like a torchsummary printout."""
    rows = []
    x = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        for i, block in enumerate(model.blocks, start=1):
            for layer in block:
                x = layer(x)
                n = sum(p.numel() for p in layer.parameters())
                rows.append((f"Block {i}: {type(layer).__name__}", tuple(x.shape[1:]), n))
        x = model.pool(x).flatten(1)
        rows.append(("Global average pool", tuple(x.shape[1:]), 0))
        x = model.classifier(x)
        rows.append(("Linear (classifier)", tuple(x.shape[1:]), sum(p.numel() for p in model.classifier.parameters())))
    return rows


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def augment(x):
    """Random flips and 90-degree rotations: a leaf is still the same leaf when turned around."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))


def evaluate(model, x, y, batch_size=128, device="cpu"):
    """Return (loss, accuracy, predictions) on tensors x, y."""
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logits.append(model(x[i:i + batch_size].to(device)).cpu())
    logits = torch.cat(logits)
    loss = nn.functional.cross_entropy(logits, y).item()
    preds = logits.argmax(1)
    return loss, (preds == y).float().mean().item(), preds.numpy()


def train_model(model, x_train, y_train, x_val, y_val, epochs=8, lr=3e-3, batch_size=32,
                use_augment=False, device="cpu", seed=0, log=None):
    """Train with Adam and return a history dict of per-epoch train/validation loss and accuracy."""
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train))
        total_loss, correct = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            if use_augment:
                xb = augment(xb)
            optimizer.zero_grad()
            out = model(xb)
            loss = nn.functional.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        val_loss, val_acc, _ = evaluate(model, x_val, y_val, device=device)
        history["train_loss"].append(total_loss / len(order))
        history["train_acc"].append(correct / len(order))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if log:
            log(f"Epoch {epoch + 1}/{epochs}: train loss {history['train_loss'][-1]:.3f}, "
                f"val acc {val_acc:.1%} ({time.time() - start:.0f} s)")
    history["seconds"] = time.time() - start
    model.cpu()
    return history
