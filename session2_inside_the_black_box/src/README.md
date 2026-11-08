# Session 2 notebooks: Inside the Black Box

Two kinds of notebooks, used together:

- **marimo notebooks (`.py`)** explain the **concepts** with sliders, buttons and plots. They are shown by the
  presenter and can be explored by anyone in the browser with **molab** (no installation, runs on CPU).
  The **Open in molab** button first shows a read-only preview with all text, figures and results (using the
  pre-trained models); to move the sliders and train the models yourself, start a session from that page.
- **Colab notebooks (`_colab.ipynb`)** are the **hands-on** part: you change a few settings, train the models
  yourself and try your own data. No coding is needed; run the cells in order with ▶.

Explore them in the order below. Each marimo notebook has a short line of text per step; click **"More details"**
to read the longer explanation. Buttons labeled **Train Model** train live (under a minute); **Load Pre-trained**
loads a model trained beforehand, which is also what each notebook shows when it opens. Each marimo notebook ends
with a **References** section listing the papers, datasets and software it builds on.

## Part 1: How neural networks learn features

**[`classification2D_synthetic.py`](classification2D_synthetic.py)** ·
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/classification2D_synthetic.py)

Colored dots that no straight line can separate. A small network learns to move the dots to new positions
(**features**) where a straight line works. Steps: the data · the network · training · looking inside the
hidden layer.

## Part 2: Convolutional networks (CNNs) — images and DNA

**[`CNN.py`](CNN.py)** ·
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/CNN.py)

Recognizing disease in bean-leaf photos (healthy, angular leaf spot, bean rust).

1. One filter by hand: slide a small filter over a leaf to find edges.
2. ReLU, pooling and the architecture; train a small CNN.
3. Look inside: the feature maps of each layer, and how much of the leaf one neuron sees (receptive field).

**Hands-on (Colab):**

- [`CNN_colab.ipynb`](CNN_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/CNN_colab.ipynb):
  train the bean-leaf CNN yourself and test it on a photo of your own leaf.
- [`DNA_motif_colab.ipynb`](DNA_motif_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/DNA_motif_colab.ipynb):
  the same idea in 1D — a filter sliding along DNA learns to find a hidden motif.

## Part 3: Recurrent networks (RNNs) — time series

**[`RNN.py`](RNN.py)** ·
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/RNN.py)

Forecasting the weather in Chachapoyas from hourly NASA POWER data.

1. Memory by hand: a one-number "memory" of recent rain.
2. The data and the forecasting task (including how the time of day is encoded).
3. Simple baselines first.
4. Train an LSTM (and see it unrolled over time).
5. Predictions, where they fail, and the hidden state.

**Hands-on (Colab):** [`RNN_colab.ipynb`](RNN_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/RNN_colab.ipynb):
train the forecaster yourself and forecast the weather at your own location.

## Part 4: Autoencoders — finding faulty sensor readings

**[`Autoencoder.py`](Autoencoder.py)** ·
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/Autoencoder.py)

The Chachapoyas weather used as a "field sensor": a network trained only on normal days flags days it cannot
rebuild.

1. A bottleneck by hand (PCA, a simple linear autoencoder).
2. The sensor data and the faults (stuck sensor, spike, drift, dropout).
3. Train the autoencoder (architecture and mirrored encoder/decoder).
4. Choose the threshold: faults found vs. false alarms.
5. The bottleneck size matters.
6. Look inside the bottleneck.

**Hands-on (Colab):** [`Autoencoder_colab.ipynb`](Autoencoder_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/Autoencoder_colab.ipynb):
detect faults with your own settings and location, and (extra) clean noisy bean-leaf images.

## Part 5: Transformers — attention, DNA and proteins

**[`Attention.py`](Attention.py)** ·
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/Attention.py)

How every token looks at every other token, and why that helps with long-range relations.

1. Attention by hand: "The animal didn't cross the street because it was too tired" (queries, keys, values,
   softmax, two heads).
2. Word order: attention alone cannot tell "dog bites man" from "man bites dog"; positional encoding
   (sine and cosine waves of different speeds) tells the model where each token is.
3. A tiny transformer on DNA: does the -35 box come before the -10 box? It succeeds where a CNN stays at chance,
   and its attention map shows how.
4. Tokens beyond words: image patches and amino acids; a protein language model (ESM-2) that learned which
   amino acids touch in the folded protein.
5. Scale: from our tiny model to today's largest models.

**Hands-on (Colab):** [`Attention_colab.ipynb`](Attention_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/Attention_colab.ipynb):
train the DNA transformer with your own motifs and settings, and run ESM-2 on any protein you choose.

## Running the marimo notebooks on your own computer

```bash
pip install marimo torch numpy pandas matplotlib scikit-learn pillow
marimo run CNN.py     # app view, as in the workshop
marimo edit CNN.py    # also shows and lets you change the code
```

Run these commands from this folder. If the data files or shared code are missing, the notebooks download them
from GitHub.

---

## Supporting files (not needed to explore the notebooks)

| File or folder | Purpose |
|---|---|
| `cnn_common.py`, `rnn_common.py`, `ae_common.py`, `attention_common.py` | Shared data loading, models and training code used by the marimo and Colab notebooks of each part. |
| [`data/`](data/) | Small datasets and pre-trained model weights loaded by the notebooks. See [`data/README.md`](data/README.md) for the dataset formats. |
| [`tools/`](tools/) | One-time scripts that built the files in `data/`. |
| `__marimo__/` | Created automatically by marimo; safe to ignore. |
