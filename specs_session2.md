# Spec: Marimo Notebooks for Session 2 — Inside the Black Box

**Workshop:** UNTRM AI Workshop, Chachapoyas, Peru · Session 2 (Monday, Nov 9, 2026)
**Author:** Edgar Lobaton · **Status:** draft for notebook generation
**Deliverable:** four marimo notebooks (Parts 2–5 of the session) that the presenter runs live in the browser (molab) while the audience watches. No participant coding.

---

## 1. Context

Session 2 follows Session 1 (“From Data to Agents”). Participants already:
- saw that images, text and DNA become tensors with shapes;
- trained a bean-leaf classifier in Colab with Gemini (TensorFlow `beans` dataset, Session 1 Activity D);
- saw Part 1 of this session: how computers learn, under/overfitting, train/validation/test, neuron → MLP (live demo: `ClassificationMLP.py`), and when deep learning beats classical ML.

**Audience:** about 15 faculty, researchers and graduate students from UNTRM. Interests: agricultural data (coffee, crops, weather), computer vision and microscopy (spores, nematodes), and genomics/biotech (the largest group: AlphaFold, proteins, gene identification). Mixed programming experience. Spanish-speaking; slides exist in English and Peruvian Spanish.

**Session plan these notebooks support** (times are for the whole part, slides + demo):

| Part | Topic | Time | Notebook |
|---|---|---|---|
| 2 | Convolutional networks: images and DNA | 20 min | `CNN.py` |
| 3 | Recurrent networks: time series | 20 min | `RNN.py` |
| 4 | Autoencoders: anomalies, denoising | 15 min | `AutoencoderAnomaly.py` (or reuse existing `Autoencoder.py`) |
| 5 | Transformers: attention, proteins, DNA | 25 min | `Attention.py` |

Each notebook is shown for about **6–10 minutes**; the rest of each part is slides.

---

## 2. Conventions for all notebooks (match the existing Part 1 demos)

Follow the style of `src-marimo/BiasVariance.py`, `ClassificationMLP.py` and `Autoencoder.py` in the GitHub repo `lobaton/DM2026-Deep-Learning-with-Python`:

- **Format:** marimo app, `app = marimo.App(width="medium")`, current marimo (the existing files were generated with 0.23.x).
- **Framework:** PyTorch, plus numpy, pandas, matplotlib, scikit-learn. No TensorFlow/Keras, so the stack matches the other demos. The 2024 Keras notebooks listed below are content sources only; port the logic to PyTorch.
- **Runs on molab on CPU:** everything, including training, must finish in **under about 60 seconds** per run on CPU. Use small subsets, small models and few epochs. Set random seeds so the live demo is repeatable.
- **Data loading:** from public URLs or synthetic generation inside the notebook. No Google Drive, no credentials, no local files. Cache downloads; show a clear message if a download fails.
- **Interactivity:** `mo.ui.slider` / `mo.ui.dropdown` for the parameters worth exploring, and `mo.ui.run_button(label="Train")` with `mo.stop(...)` so training only runs on click (as in `Autoencoder.py`).
- **Layout:** helper functions in a `hide_code=True` cell at the top; short `mo.md` explanations (2–4 sentences) before each step; code cells visible but short. Use `mo.hstack`/`mo.vstack` to put controls next to plots.
- **Projection-friendly plots:** large fonts (≥14 pt), thick lines, at most 2–3 panels per figure, colorblind-safe colors, axis labels with units.
- **Honest evaluation:** always show a held-out test result and, where it makes sense, a simple baseline (for example, persistence for forecasting). This reinforces Part 1 and Session 1’s evaluation checklist.
- **Domain framing:** examples and wording from agriculture, microscopy and genomics, not generic engineering.
- **Ending:** each notebook ends with a “What to notice” cell (3 bullets) and 2 discussion questions, in the style of the orange questions in the 2024 notebooks.
- **Language:** English text in the notebook. Keep strings simple so a Spanish version can be produced later. Keep technical terms (dataset, epoch, token, embedding, attention) in English.
- **Files and links:** save to `src-marimo/` in the workshop repo. molab link pattern: `https://molab.marimo.io/github/<owner>/<repo>/blob/main/src-marimo/<File>.py`.

---

## 3. Notebook specs

### 3.1 `CNN.py` — Convolutional networks (Part 2)

**Learning goals**
1. A convolution is a small filter sliding over the data that responds to a local pattern.
2. A CNN learns its own filters; stacked layers build edges → textures → shapes.
3. The same idea works in 1D: a filter sliding along DNA finds motifs.

**Flow**
1. **One filter by hand (2D).** Load a single bean-leaf photo. Dropdown of classic kernels (vertical edge, horizontal edge, blur, sharpen) → show input and feature map side by side. Slider for kernel size or stride is optional.
2. **Train a tiny CNN on bean leaves.** Use the `beans` dataset (3 classes: healthy, angular leaf spot, bean rust), downsampled to 64×64, with a few hundred training images so it trains in under 60 s on CPU. Two conv blocks plus a linear head. Run button → learning curve, test accuracy and confusion matrix.
   - **Data loading:** avoid `tensorflow_datasets` (too heavy for molab). Use the Hugging Face copy `AI-Lab-Makerere/beans` (parquet), or pre-downsample once and host a small `.npz` in the repo. Verify which works in molab.
3. **Look inside.** Show the learned first-layer filters and the feature maps for one leaf at layer 1 and layer 2. Connect to “the network learned its own features” from Part 1.
4. **1D CNN on DNA (genomics hook).** Synthetic task: random DNA sequences of length 100, one-hot encoded (shape `(4, L)`, matching the Session 1 tensor slide). Positives contain a planted motif (for example `TATAAT`) at a random position, plus point mutations. Train a 1D CNN (one conv layer with a few filters + global max pool). Show the learned filter as a sequence logo or heatmap that recovers the motif, and a per-position activation track highlighting where the motif sits.
5. **What to notice / questions.** For example: why might a model trained on lab photos fail on field photos? What would a filter look for in a spore image?

**Interactive controls:** kernel dropdown; number of training images; number of filters; Train button.
**Source material:** BasicArchitectures (CNN section, slides 1–12); ECE 525 Set05 `Script - CIFAR10 Classification.ipynb` for the PyTorch training loop; Session 1 Activity D (beans).
**Complementary external demo:** CNN Explainer (Georgia Tech, poloclub.github.io/cnn-explainer) as a backup.

---

### 3.2 `RNN.py` — Recurrent networks and time series (Part 3)

**Learning goals**
1. An RNN carries a hidden state from one time step to the next; an LSTM uses gates to remember longer.
2. Forecasting framing: a window of past readings → a value some hours ahead.
3. Always compare with a naive baseline.

**Flow**
1. **Data.** Weather-station series. Default: the Jena climate dataset (Max Planck Institute for Biogeochemistry, 14 variables every 10 minutes, 2009–2016), downloaded from `https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip`. Resample to hourly. Use 1–2 years only for speed.
   - **Preferred if available:** a local UNTRM / Amazonas weather-station series (ask Jorge or Miguel). Keep the loader generic so a CSV with a timestamp column and variables like temperature, relative humidity and rainfall can be swapped in.
2. **Explore.** Plot temperature and humidity over a few weeks. Explain the window/target construction with a small diagram (past `H` hours → value at `+F` hours).
3. **Baselines first.** Persistence (“tomorrow = today”) and a seasonal baseline (“same hour yesterday”). Report MAE in °C.
4. **Train an LSTM.** Sliders: history length `H` (24–120 h), horizon `F` (1–24 h), hidden size. Run button → train/validation loss curve and test MAE next to the baselines.
5. **Predictions.** Plot several test windows (history, true future, prediction). Show where it fails, such as sudden fronts.
6. **Link to use cases.** Humidity and temperature drive coffee leaf rust risk (callback to the Session 1 orchestration example); soil moisture for irrigation; sensor streams (mussel and wearable work).
7. **What to notice / questions.** For example: does a longer history always help? When does the LSTM beat persistence?

**Interactive controls:** history length, horizon, hidden size, variable selection (multiselect), Train button.
**Source material:** 2024 Data Matters `04 - Weather Forecasting - RNN.ipynb` (Keras LSTM on Jena, 7 features, 120 h history, 12 h horizon: port to PyTorch); ECE 525 Set05 `Script - RNN.ipynb` (PyTorch RNN cell from scratch, a good model for the “hidden state” cell code); BasicArchitectures (RNN section, slides 13–20).

---

### 3.3 `AutoencoderAnomaly.py` — Autoencoders (Part 4)

**Option A (fastest):** reuse the existing `src-marimo/Autoencoder.py` (MIT-BIH ECG, `wfdb`, latent-dimension slider, Train button, ROC/AUC, t-SNE). Only add a short intro cell framing it for sensors in the field.

**Option B (recommended for this audience):** a new, lighter notebook on sensor-like time series.

**Learning goals**
1. Encoder → bottleneck → decoder, trained to reconstruct its own input.
2. Train on normal data only; a high reconstruction error flags an anomaly.
3. The bottleneck size matters: too big and the network just copies.

**Flow**
1. **Data.** Default: Numenta Anomaly Benchmark files `artificialNoAnomaly/art_daily_small_noise.csv` (train) and `artificialWithAnomaly/art_daily_jumpsup.csv` (test), from `https://raw.githubusercontent.com/numenta/NAB/master/data/`. Frame them as “a daily sensor cycle, like a soil-moisture or greenhouse sensor.”
   - **Alternative:** reuse the weather series from `RNN.py` with injected faults (stuck sensor, spike, drift), selectable from a dropdown. This connects Parts 3 and 4.
2. **Model.** 1D convolutional autoencoder on windows of one day (288 steps at 5-minute sampling). Slider for latent size.
3. **Train (run button)** → loss curve; overlay an original window and its reconstruction.
4. **Threshold.** Histogram of training reconstruction errors with a threshold slider (default: maximum or 99th percentile). Mark flagged points in red on the test series.
5. **Bottleneck matters.** Rerun with a large latent size and show that anomalies are reconstructed too, so detection fails.
6. **Optional (if time):** denoising — add noise to bean-leaf or microscopy patches and reconstruct, as a hook to microscopy denoising.
7. **What to notice / questions.** For example: how would you choose the threshold in a real deployment? What does a false alarm cost in the field?

**Interactive controls:** latent size, threshold, fault type (option B alternative), Train button.
**Source material:** 2024 Data Matters `05 - Time Series Anomaly Detection - AE.ipynb` (Keras Conv1D AE on NAB: port to PyTorch); ECE 525 Set05 `Script - AEs for Anomaly Detection.ipynb` and `helper_fncs.py`; ECE 525 `Anomaly in ECG using AE.ipynb`; BasicArchitectures (AE section, slides 21–28).

---

### 3.4 `Attention.py` — Transformers: the basics (Part 5)

**Scope note:** Session 3 (Tuesday) covers using pre-trained transformers in code, plus diffusion. Session 2 covers concepts only: attention, tokens beyond words, and why pre-training matters. No diffusion here.

**Learning goals**
1. Self-attention: every token scores every other token (query · key), the softmax turns scores into weights, and the output is a weighted mix of values.
2. Transformers process the whole sequence in parallel and capture long-range context (contrast with the RNN from Part 3).
3. Tokens can be words, image patches, DNA k-mers or amino acids. Pre-trained models (ESM, DNABERT, Vision Transformers) are trained once on huge data and then adapted.

**Flow**
1. **Attention by hand.** A short sentence, for example the classic “The animal didn’t cross the street because it was too tired” from the slides, with tiny 2–4-D toy embeddings. Show the Q·Kᵀ score matrix, the softmax weights as a heatmap, and the output. A temperature or scale slider shows how attention sharpens or spreads. Use the Session 1 tensor-shape notation.
2. **Positional information.** Shuffle the tokens and show that attention alone is order-blind; add positional encodings and show the difference. Optionally plot the sinusoidal encodings.
3. **A tiny transformer on DNA (genomics hook, mirrors CNN.py step 4).** Synthetic classification where the label depends on two motifs being present far apart, which needs long-range context. Train a one-layer, one- or two-head transformer encoder over k-mer or nucleotide tokens in under 60 s. Show test accuracy against the 1D CNN from Part 2 trained on the same task, and the attention map for one sequence, where the motif positions attend to each other.
4. **Optional, real model (verify it runs in molab first):** ESM-2 smallest checkpoint (`facebook/esm2_t6_8M_UR50D`, about 8M parameters) via `transformers`. Paste a short protein sequence and show one layer’s attention map, and optionally the predicted contact map, as the bridge to AlphaFold. If download or runtime is a problem on molab, replace it with a precomputed attention map saved as `.npy` in the repo.
5. **Scale.** A small table of parameter counts (toy model in this notebook → ESM-2 8M → GPT-2 → current LLMs), matching the “Scale of state-of-the-art models” slide.
6. **What to notice / questions.** For example: what is a “token” in your data? Why can a transformer relate two distant motifs more easily than an RNN?

**Interactive controls:** sentence or sequence input (`mo.ui.text`), temperature slider, head selector, positional-encoding toggle, Train button.
**Source material:** AdvancedArchitectures (Transformers, slides 1–12: self-attention worked example, Q/K/V, multi-head, positional encoding, embedding, end-to-end pipeline, scale); 2024 Data Matters `08 - Text Classification - Transformer.ipynb` (Keras transformer block and token + position embedding: port the block to PyTorch); ECE 525 `RestNet and Transformer` slides (Set13); session-1 slides on next-token prediction and tensors (to keep notation consistent).
**Complementary external demo:** Transformer Explainer (Georgia Tech, poloclub.github.io/transformer-explainer), which runs GPT-2 live in the browser.

---

## 4. Resource locations (Google Drive and GitHub)

### Workshop folder
| Resource | Location |
|---|---|
| Session descriptions (`Notes.txt`) | Drive, shared drive, folder “2026m11 - Workshop - AI - Peru”: https://drive.google.com/file/d/1OvmwtehPIsuDWsA75DE3EzrZ_jg9bSjZ/view |
| Session 2 deck (opening + Part 1, in progress) | https://claude.ai/artifact/P9Z1DU3kb5kmWMu2Ujmehr (private; share from the deck if needed) |

### Slides to align with (summer 2026 Data Matters, “Deep Learning with Python”)
Drive path: *Data Matters - Summer Workshop / 2026 - Summer Workshop / Deep Learning with Python* (https://drive.google.com/drive/folders/1uL5b0y-yrHNLdu6HudIcdgPxcr3Q8Auu)

| File | Use | Link |
|---|---|---|
| Day1 / `BasicArchitectures.pptx` | CNN (1–12), RNN (13–20), AE (21–28) | https://drive.google.com/file/d/1KPs4A2DP5qwZZ0fPlT7u5p2BZg8VzMj2/view |
| Day2 / `AdvancedArchitectures.pptx` | Transformers (1–12), diffusion (13–20, Session 3 only) | https://drive.google.com/file/d/1SldNIEq_H2LSQd7du6ajlOLtnIbe0gmr/view |
| Day1 / `FromClasssical2DL.pptx` | Part 1 source (already in the deck) | https://drive.google.com/file/d/1WMPkuFrtqUIN3YKWZr4XGI6FLds_OjS6/view |
| Day2 / `BuildingACNN.pptx` | Example of a CNN built step by step | https://drive.google.com/file/d/1AjdeiC4WoU6ctdPkk9gbSxk4EyEIVEpP/view |
| Day2 / `TransferLearning.pptx` | Bridge to Session 3 | https://drive.google.com/file/d/1ki1ECanbltw_9kHv-atD4oumGpgmPOWa/view |
| Data Matters “AI and Deep Learning” / `Architectures.pptx` | Shorter combined CNN/RNN/AE/Transformer deck | https://drive.google.com/file/d/16SQo6S7_5n-Yjezv-awC3IXCvzKhNSCo/view |

The same slides as PDFs are in the GitHub repos: `lobaton/DM2026-Deep-Learning-with-Python/Slides/` (BasicArchitectures.pdf, AdvancedArchitectures.pdf, BuildingACNN.pdf, TransferLearning.pdf) and `lobaton/DM2026-AI-and-Deep-Learning/Slides/Architectures.pdf`.

### Existing marimo demos (GitHub, style reference)
`lobaton/DM2026-Deep-Learning-with-Python/src-marimo/`: `BiasVariance.py`, `LogisticRegression.py`, `ClassificationMLP.py`, `Autoencoder.py` (ECG anomaly detection with MIT-BIH via `wfdb`).

### ECE 525 (Neural Networks and Deep Learning), shared drive
Folder *NeuralNets-CurrentSlides / Set05 - Basic Architectures*: https://drive.google.com/drive/folders/19DxElkRA6QSNnrbYEsrYRn-AuFEFJypk

| File | Use | Link |
|---|---|---|
| `Basic Architectures 1 - CNN.pptx` | Longer CNN lecture | https://drive.google.com/file/d/1LSQYbvQZ0Uz1AA3NmD9Wjo8pcbBdVoTB/view |
| `Basic Architectures 2 - AE.pptx` | Longer AE lecture | https://drive.google.com/file/d/1MDfh6n-H9TJqESmvfRBQQh11w3cobeBw/view |
| `Basic Architectures 3 - RNN.pptx` | Longer RNN/LSTM lecture | https://drive.google.com/file/d/1A6Ua6oBG-4xlQcKVSYEP0k-oEmmuaPFx/view |
| `Script - CIFAR10 Classification.ipynb` | PyTorch CNN training loop | https://colab.research.google.com/drive/19JLDlb3GjMFjaPgNmtLa8VoSzNUFc6At |
| `Script - RNN.ipynb` | PyTorch RNN cell from scratch (name classification) | https://colab.research.google.com/drive/1NI0EAsWidRTGLz4Z1lqzTcijyFwxCD01 |
| `Script - AEs for Anomaly Detection.ipynb` | PyTorch AE anomaly detection | https://colab.research.google.com/drive/1h9kqMnXn-NzWHeEvE_qtZyjgLQkDgIqh |
| `helper_fncs.py` | Helpers used by the AE script | https://drive.google.com/file/d/130-IAZ-du3kmFkqwyyxbpcwo_TGbmcJF/view |
| Google Forms “Interact - CNN / AE / RNN” | Quiz questions to reuse as show-of-hands | in the same folder |

Other ECE 525 folders: *Set13 - Residual Networks and Transformers* (`RestNet and Transformer.pptx`: https://drive.google.com/file/d/1r3fWkOibkvHVg5Ht15IYzc0Hfb6AVcvx/view) and *Set10 - VAE* (https://drive.google.com/drive/folders/1EM4vH2EiUxj0acDKBGgSfPXiVZybZAXA).

### 2024 Data Matters exercises (Keras; port the logic to PyTorch)
Folder *Minicourse - Data Matters - Deep Learning in Python / Exercises*: https://drive.google.com/drive/folders/13KABPA1Jgn6wL2Q0_1IgC6dQnf_zqyxc

| File | Use | Link |
|---|---|---|
| `04 - Weather Forecasting - RNN.ipynb` | Jena climate LSTM forecasting → `RNN.py` | https://colab.research.google.com/drive/1vgyzp89JiF4oYaAnEANxdDncr9AA_InJ |
| `05 - Time Series Anomaly Detection - AE.ipynb` | NAB Conv1D AE → `AutoencoderAnomaly.py` | https://colab.research.google.com/drive/19Ten2QtfoY27QjQFOG6qwbrnKTOSavuo |
| `08 - Text Classification - Transformer.ipynb` | Transformer block + token/position embedding → `Attention.py` | https://colab.research.google.com/drive/1okhj3lpjzomKQy7Jm2WdEXIr986isKXM |
| `06 - Image Classification - Transfer Learning.ipynb` | Session 3 bridge | https://colab.research.google.com/drive/1-ZPa6CVNj6ThQt9mRGcTr-QTeX23mGFs |

### Use-case papers for slides (nematodes and microscopy, shared drive)
- ViT ensemble for nematode classification: https://drive.google.com/file/d/18Wv5ID8lJD_T6WTdKOHddsrGgZQ7fUI0/view
- WormSwin, C. elegans instance segmentation with a vision transformer: https://drive.google.com/file/d/1VNLTjBZadP_DNO0c6rQzbXk6FU23uty8/view
- CNN and RNN for live/dead C. elegans classification: https://drive.google.com/file/d/1nvrTYhhSAvJl6zU-Iq6K5YkJcLXAUfjZ/view
- Mask R-CNN nematode segmentation: https://drive.google.com/file/d/1-GiSZLXK6F0aHpJqJciyj5WHQS0OJl74/view

### Public datasets (no credentials)
| Dataset | Notebook | Source |
|---|---|---|
| Beans (Makerere AI Lab), 3 classes, 1,295 images | `CNN.py` | Hugging Face `AI-Lab-Makerere/beans`; TFDS catalog: tensorflow.org/datasets/catalog/beans |
| Synthetic DNA with planted motifs | `CNN.py`, `Attention.py` | generated in the notebook |
| Jena climate 2009–2016 | `RNN.py`, `AutoencoderAnomaly.py` (alt.) | storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip |
| Numenta Anomaly Benchmark (artificial daily series) | `AutoencoderAnomaly.py` | raw.githubusercontent.com/numenta/NAB/master/data/ |
| MIT-BIH Arrhythmia (via `wfdb`) | existing `Autoencoder.py` | physionet.org/content/mitdb/1.0.0/ |
| ESM-2 8M checkpoint (optional) | `Attention.py` | Hugging Face `facebook/esm2_t6_8M_UR50D` |

---

## 5. Acceptance checklist (per notebook)

- [ ] Opens and runs top to bottom on molab with no edits and no credentials.
- [ ] Each Train click finishes in under about 60 s on CPU; results are repeatable with the default seed.
- [ ] At least one held-out test metric is shown; forecasting also shows baselines.
- [ ] All controls update their plots without rerunning training unnecessarily.
- [ ] Plots are readable from the back of a room (fonts ≥14 pt).
- [ ] Ends with “What to notice” and two discussion questions.
- [ ] A molab link is added to the Session 2 slides (one red “Live demo” slide per notebook, like the Part 1 MLP demo).

## 6. Open decisions

1. Local data: can UNTRM provide a weather-station series (and, ideally, a few spore or nematode images) to replace Jena and beans?
2. Autoencoder: reuse the existing ECG notebook (option A) or build the sensor version (option B)?
3. Transformers: include the optional ESM-2 step, or keep everything synthetic?
4. Should Spanish versions of the notebooks be produced, or English notebooks with Spanish slides?
