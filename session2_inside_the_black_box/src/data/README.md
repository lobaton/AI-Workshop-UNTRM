# `beans128.npz`: bean-leaf disease images, 128×128

A compact copy of the **beans** dataset (Makerere AI Lab, Uganda; Hugging Face
`AI-Lab-Makerere/beans`), resized to 128×128 RGB so it loads fast and trains on a CPU. Built by
`../tools/prep_beans.py`.

- **Task:** 3-class image classification of bean leaves photographed in the field with a phone.
- **Images:** 1,295 in total, each a 128×128 RGB JPEG (quality 90), resized from the original 500×500
  with Lanczos filtering (no cropping, so the aspect ratio is unchanged).
- **File size:** about 10.7 MB.
- **Download URL:**
  `https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/session2_inside_the_black_box/src/data/beans128.npz`

## Arrays in the file

The images are stored as JPEG bytes, not as a pixel array. All JPEG files are concatenated into one
byte array, and `offsets` marks where each one starts.

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `jpeg` | `uint8` | `(10630842,)` | All JPEG files, concatenated byte by byte |
| `offsets` | `int64` | `(1296,)` | Image `i` is `jpeg[offsets[i]:offsets[i+1]]`; `offsets[0] = 0`, `offsets[-1] = len(jpeg)` |
| `labels` | `int64` | `(1295,)` | Class of each image (see below) |
| `split` | `int64` | `(1295,)` | Which split each image belongs to (see below) |

Index `i` refers to the same image in `labels`, `split` and `offsets`. The file contains no Python
objects, so `np.load` works without `allow_pickle`.

## Labels

| `labels` value | Class name | Meaning |
|---|---|---|
| `0` | `angular_leaf_spot` | Angular leaf spot (fungal disease, *Pseudocercospora griseola*) |
| `1` | `bean_rust` | Bean rust (fungal disease, *Uromyces appendiculatus*) |
| `2` | `healthy` | Healthy leaf |

The label order is the same as in the original Hugging Face dataset.

## Splits

| `split` value | Split | Images | angular_leaf_spot / bean_rust / healthy |
|---|---|---|---|
| `0` | train | 1,034 | 345 / 348 / 341 |
| `1` | validation | 133 | 44 / 45 / 44 |
| `2` | test | 128 | 43 / 43 / 42 |

These are the original Hugging Face splits. Images are stored in split order (all train, then validation,
then test). The classes are balanced, so chance accuracy is about 33%.

## Loading example (Python)

```python
import io
import urllib.request
import numpy as np
from PIL import Image

URL = ("https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
       "session2_inside_the_black_box/src/data/beans128.npz")
urllib.request.urlretrieve(URL, "beans128.npz")

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
SPLITS = {"train": 0, "validation": 1, "test": 2}

pack = np.load("beans128.npz")
jpeg, offsets = pack["jpeg"], pack["offsets"]

# Decode every image to a uint8 array of shape (N, 128, 128, 3), values 0-255, RGB channel order
images = np.stack([
    np.asarray(Image.open(io.BytesIO(jpeg[offsets[i]:offsets[i + 1]].tobytes())).convert("RGB"))
    for i in range(len(offsets) - 1)
])
labels, split = pack["labels"], pack["split"]

x_train, y_train = images[split == 0], labels[split == 0]   # (1034, 128, 128, 3), (1034,)
x_val, y_val = images[split == 1], labels[split == 1]       # (133, 128, 128, 3), (133,)
x_test, y_test = images[split == 2], labels[split == 2]     # (128, 128, 128, 3), (128,)
```

Decoding all 1,295 images takes under a second. The images are channels-last `(H, W, C)` `uint8`.

- **PyTorch:** convert to `float32`, divide by 255 and permute to `(N, 3, 128, 128)`.
- **TensorFlow/Keras:** keep channels-last.

## Source and license

Original dataset: Makerere AI Lab, *Bean disease dataset* (2020), distributed on Hugging Face as
`AI-Lab-Makerere/beans` and in TensorFlow Datasets as `beans`. Check the original dataset page for its
license terms before redistributing.

---

# `chachapoyas_hourly.csv`: hourly weather for Chachapoyas, Peru (2023–2024)

Hourly weather estimates for Chachapoyas, Amazonas, Peru (latitude −6.23, longitude −77.87), downloaded from
**NASA POWER** (community `AG`). Built by `../tools/prep_weather.py`.

- **Task:** time-series forecasting (for example, temperature or humidity a few hours ahead).
- **Rows:** 17,544, one per hour from `2023-01-01 00:00` to `2024-12-31 23:00`. There are no gaps and no
  missing values (NASA's `-999` fill values were interpolated; none occurred in this period).
- **File size:** about 0.8 MB, plain CSV with a header row.
- **Download URL:**
  `https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/session2_inside_the_black_box/src/data/chachapoyas_hourly.csv`
- **Important:** NASA POWER values come from satellites and weather models on a grid of roughly 50 km. They are
  not weather-station measurements, so local effects (valleys, slopes) are smoothed out.

## Columns

| Column | Type | Unit | NASA POWER parameter | Meaning |
|---|---|---|---|---|
| `time` | text `YYYY-MM-DD HH:MM` | local solar time | – | Start of the hour |
| `temperature` | float | °C | `T2M` | Air temperature at 2 m |
| `humidity` | float | % | `RH2M` | Relative humidity at 2 m (0–100) |
| `rain` | float | mm/h | `PRECTOTCORR` | Precipitation (bias-corrected) |
| `wind` | float | m/s | `WS2M` | Wind speed at 2 m |
| `pressure` | float | kPa | `PS` | Surface pressure (about 79 kPa because of the altitude; NASA's grid-cell elevation is 2,112 m) |

Time is **local solar time** (NASA POWER `time-standard=LST`), which at this longitude is about 11 minutes
behind Peru time (UTC−5). Values are rounded to 2 decimals. There are no labels: the target is chosen
from the columns (the workshop notebooks forecast `temperature` or `humidity`).

## Splits used in the workshop

The splits are **chronological, with no shuffling**: rows are used in time order so the model is always tested on
a period after the one it learned from.

| Split | Rows (0-based, end exclusive) | Period |
|---|---|---|
| train | 0 – 12,280 (70%) | 2023-01-01 00:00 to 2024-05-26 15:00 |
| validation | 12,280 – 14,912 (15%) | 2024-05-26 16:00 to 2024-09-13 07:00 |
| test | 14,912 – 17,544 (15%) | 2024-09-13 08:00 to 2024-12-31 23:00 |

## Loading example (Python)

```python
import pandas as pd

URL = ("https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
       "session2_inside_the_black_box/src/data/chachapoyas_hourly.csv")
df = pd.read_csv(URL, parse_dates=["time"])   # (17544, 6)

n = len(df)
train = df.iloc[: int(0.70 * n)]
val = df.iloc[int(0.70 * n): int(0.85 * n)]
test = df.iloc[int(0.85 * n):]
```

A typical forecasting setup builds windows from these rows: the past `H` hours of the chosen columns as
input and the target column `F` hours after the last input hour as output. Normalize each column with
the mean and standard deviation of the **training** rows only. Useful baselines are persistence (the value
at the last input hour) and the value at the same hour on the previous day. Temperature and humidity have
a strong daily cycle, so adding the hour of day as `sin(2π·hour/24)` and `cos(2π·hour/24)` helps.

## Source and license

NASA POWER (Prediction Of Worldwide Energy Resources), NASA Langley Research Center,
<https://power.larc.nasa.gov/>, hourly API, community `AG`. NASA POWER data are freely available; please
acknowledge NASA POWER when using them.

---

# `esm2_ubiquitin.npz`: ESM-2 attention and contact maps for ubiquitin

Precomputed outputs of three sizes of the **ESM-2** protein language model on **human ubiquitin** (76 amino
acids), together with the true contact map from its crystal structure. Built by `../tools/precompute_esm.py`,
so the notebooks can show them without the `transformers` package or a model download.

- **File size:** about 65 kB.
- **Download URL:**
  `https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/session2_inside_the_black_box/src/data/esm2_ubiquitin.npz`

## Arrays in the file

`M = 3` models, `L = 76` amino acids. Model index 0, 1, 2 = 8M, 35M and 150M parameters.

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `sequence` | `str` | `()` | The protein sequence, one-letter amino-acid code |
| `models` | `str` | `(M,)` | Hugging Face model names (`facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D`, `facebook/esm2_t30_150M_UR50D`) |
| `params` | `int64` | `(M,)` | Number of parameters of each model |
| `contacts` | `float16` | `(M, L, L)` | Predicted contact probability for every pair of positions (symmetric, 0–1) |
| `true_contacts` | `uint8` | `(L, L)` | 1 if the two residues' Cβ atoms (Cα for glycine) are closer than 8 Å in PDB 1UBQ |
| `precision` | `float64` | `(M,)` | Share of each model's top-L predicted pairs (at least 6 positions apart) that are true contacts |
| `best_head` | `int64` | `(M, 2)` | 0-based (layer, head) of the attention head whose map best matches the true contacts |
| `best_head_attention` | `float16` | `(M, L, L)` | That head's attention weights (row = position that looks, column = position looked at; rows sum to about 1) |

Positions are 0-based (index 0 = the first amino acid, M). The start and end tokens of the ESM tokenizer are
removed. The file contains no Python objects, so `np.load` works without `allow_pickle`.

| Model | Parameters | Top-L contact precision |
|---|---|---|
| ESM-2 8M | 7.5 million | 34% |
| ESM-2 35M | 33.5 million | 66% |
| ESM-2 150M | 148 million | 78% |

## Loading example (Python)

```python
import io, urllib.request
import numpy as np

URL = ("https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
       "session2_inside_the_black_box/src/data/esm2_ubiquitin.npz")
esm = np.load(io.BytesIO(urllib.request.urlopen(URL).read()))
print(str(esm["sequence"]), esm["precision"])
predicted = esm["contacts"][2].astype(float)   # 150M model, (76, 76)
true = esm["true_contacts"]                   # (76, 76)
```

## Source and license

ESM-2 models: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language
model", *Science* 379 (2023); Meta AI, MIT license, via Hugging Face. Structure: PDB entry 1UBQ (Vijay-Kumar,
Bugg and Cook, 1987), RCSB Protein Data Bank, data available under CC0.
