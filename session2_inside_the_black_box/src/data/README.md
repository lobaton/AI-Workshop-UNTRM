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
