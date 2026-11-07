"""Shared code for the Session 2 transformer notebooks (marimo Attention.py, Attention_colab.ipynb,
tools/pretrain_attention.py).

Synthetic DNA task: every sequence contains two bacterial promoter boxes, the -35 box (MOTIF_A) and the -10 box
(MOTIF_B), far apart. The label says whether they are in promoter order (A before B) or reversed. A small CNN
with global max pooling cannot tell the order; a one-layer transformer with positional encoding can.
"""
import os
import time
import urllib.request

import numpy as np
import torch
from torch import nn

REPO_RAW = "https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/session2_inside_the_black_box/src"
WEIGHTS_FILE = "data/dna_transformer.pt"
ESM_FILE = "data/esm2_ubiquitin.npz"

BASES = "ACGT"
MOTIF_A = "TTGACA"  # -35 box
MOTIF_B = "TATAAT"  # -10 box
K = 6  # k-mer token length (as in DNABERT)


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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def to_text(seq):
    return "".join(BASES[b] for b in seq)


def make_sequences(n, length=64, min_gap=10, max_dist=30, mutation=0.0, seed=0,
                   motif_a=MOTIF_A, motif_b=MOTIF_B):
    """n random sequences, each with motif_a and motif_b planted once, separated by at least `min_gap` letters
    and starting at most `max_dist` letters apart. Half are in order A -> B (label 1), half B -> A (label 0).

    Returns (seqs int64 (n, length) with 0-3 = A, C, G, T; labels (n,); starts (n, 2) of A and B).
    """
    rng = np.random.default_rng(seed)
    la, lb = len(motif_a), len(motif_b)
    lo_dist = max(la, lb) + min_gap
    seqs = np.zeros((n, length), dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    starts = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        label = int(i % 2 == 0)
        while True:
            s = rng.integers(0, 4, length)
            dist = int(rng.integers(lo_dist, max_dist + 1))
            first = int(rng.integers(0, length - dist - max(la, lb) + 1))
            pa, pb = (first, first + dist) if label else (first + dist, first)
            for motif, p in ((motif_a, pa), (motif_b, pb)):
                for j, c in enumerate(motif):
                    s[p + j] = rng.integers(0, 4) if rng.random() < mutation else BASES.index(c)
            text = to_text(s)
            # Keep only sequences where each motif appears exactly once (no accidental extra copies)
            if mutation > 0 or (text.count(motif_a) == 1 and text.count(motif_b) == 1):
                break
        seqs[i], labels[i], starts[i] = s, label, (pa, pb)
    order = rng.permutation(n)
    return seqs[order], labels[order], starts[order]


def make_dataset(n_train=3000, n_val=500, n_test=1000, seed=0, **kwargs):
    """Train / validation / test splits, each a (seqs, labels, starts) tuple."""
    return {name: make_sequences(n, seed=seed + k, **kwargs)
            for k, (name, n) in enumerate([("train", n_train), ("validation", n_val), ("test", n_test)])}


def kmer_names(seq, k=K):
    """The overlapping k-mer tokens of one sequence, as text."""
    text = to_text(seq)
    return [text[i:i + k] for i in range(len(text) - k + 1)]


# ---------------------------------------------------------------------------
# Models (both take integer letter sequences, shape (batch, length))
# ---------------------------------------------------------------------------
def sinusoidal_encoding(n_positions, d):
    """Positional encoding of the original transformer: sine/cosine pairs turning at different speeds."""
    pos = np.arange(n_positions)[:, None]
    freq = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    pe = np.zeros((n_positions, d), dtype=np.float32)
    pe[:, 0::2], pe[:, 1::2] = np.sin(pos * freq), np.cos(pos * freq)
    return torch.from_numpy(pe)


class TinyTransformer(nn.Module):
    """k-mer tokens -> embedding (+ positional encoding) -> one transformer layer -> average -> 2 classes."""

    def __init__(self, length=64, d=32, heads=2, use_position=True, k=K):
        super().__init__()
        self.k, self.use_position = k, use_position
        self.register_buffer("kmer_weights", 4 ** torch.arange(k - 1, -1, -1))
        self.embed = nn.Embedding(4 ** k, d)
        self.register_buffer("position", sinusoidal_encoding(length - k + 1, d))
        self.attention = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.feed_forward = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, 2)

    def tokens(self, x):
        """Letters (batch, length) -> overlapping k-mer ids (batch, length - k + 1)."""
        return (x.unfold(1, self.k, 1) * self.kmer_weights).sum(-1)

    def forward(self, x, return_attention=False):
        h = self.embed(self.tokens(x))
        if self.use_position:
            h = h + self.position
        a, weights = self.attention(h, h, h, need_weights=return_attention, average_attn_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.feed_forward(h))
        logits = self.classifier(h.mean(dim=1))
        return (logits, weights) if return_attention else logits


class MotifCNN(nn.Module):
    """The 1D CNN of DNA_motif_colab.ipynb: filters slide along the letters, global max pooling, linear layer."""

    def __init__(self, n_filters=4, kernel_size=8):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, kernel_size)
        self.classifier = nn.Linear(n_filters, 2)

    def forward(self, x):
        onehot = nn.functional.one_hot(x, 4).float().transpose(1, 2)  # (batch, 4, length)
        return self.classifier(torch.relu(self.conv(onehot)).amax(dim=2))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def layer_table(model, length=64):
    """Rows of (layer, output shape, parameters) for a TinyTransformer."""
    x = torch.zeros(1, length, dtype=torch.long)
    n_tok, d = length - model.k + 1, model.embed.embedding_dim
    rows = [
        (f"Input: DNA letters", (length,), 0),
        (f"{model.k}-mer tokens (overlapping)", (n_tok,), 0),
        (f"Embedding ({4 ** model.k:,} possible tokens → {d} numbers)", (n_tok, d), count_params(model.embed)),
        ("+ positional encoding (sine/cosine, fixed)" if model.use_position else "(no positional encoding)",
         (n_tok, d), 0),
        (f"Self-attention ({model.attention.num_heads} heads) + add & normalize", (n_tok, d),
         count_params(model.attention) + count_params(model.norm1)),
        ("Feed-forward (two Linear layers, ReLU) + add & normalize", (n_tok, d),
         count_params(model.feed_forward) + count_params(model.norm2)),
        ("Average over tokens", (d,), 0),
        ("Linear (classifier)", (2,), count_params(model.classifier)),
    ]
    with torch.no_grad():
        assert model(x).shape == (1, 2)
    return rows


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def predict(model, x, batch_size=1000):
    """Probability of label 1 (A before B) for integer sequences x."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out.append(torch.softmax(model(torch.as_tensor(x[i:i + batch_size])), dim=1)[:, 1].numpy())
    return np.concatenate(out)


def accuracy(model, x, y):
    return float(((predict(model, x) > 0.5) == np.asarray(y)).mean())


def attention_maps(model, seq):
    """Attention weights of one sequence: (heads, tokens, tokens); row = token that looks, column = looked at."""
    model.eval()
    with torch.no_grad():
        _, weights = model(torch.as_tensor(seq)[None], return_attention=True)
    return weights[0].numpy()


def train_model(model, x, y, x_val, y_val, epochs=20, lr=3e-3, batch_size=64, seed=0, log=None):
    """Train with cross-entropy and Adam. History holds per-epoch training loss and validation accuracy."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = torch.as_tensor(x), torch.as_tensor(y)
    history = {"train_loss": [], "val_acc": []}
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(y)).split(batch_size):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history["train_loss"].append(total / len(y))
        history["val_acc"].append(accuracy(model, x_val, y_val))
        if log:
            log(f"Epoch {epoch + 1}/{epochs}: loss {history['train_loss'][-1]:.3f}, "
                f"validation accuracy {history['val_acc'][-1]:.1%} ({time.time() - start:.0f} s)")
    history["seconds"] = time.time() - start
    return history
