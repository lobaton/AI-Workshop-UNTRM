"""One-time prep: download the Hugging Face beans dataset and save a small 128x128 JPEG pack.

Usage (from session2_inside_the_black_box/src):
    python tools/prep_beans.py [--size 128] [--quality 90] [--cache DIR]

Output: data/beans128.npz with arrays
    jpeg    uint8, all JPEG files concatenated
    offsets int64, start of each image in `jpeg` (length N+1)
    labels  int64, 0 = angular leaf spot, 1 = bean rust, 2 = healthy
    split   int64, 0 = train, 1 = validation, 2 = test
"""
import argparse
import io
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HF_URL = "https://huggingface.co/api/datasets/AI-Lab-Makerere/beans/parquet/default/{split}/0.parquet"
SPLITS = ["train", "validation", "test"]

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--quality", type=int, default=90)
parser.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "beans_parquet"))
args = parser.parse_args()

os.makedirs(args.cache, exist_ok=True)
chunks, offsets, labels, split_ids = [], [0], [], []
for split_id, split in enumerate(SPLITS):
    path = os.path.join(args.cache, f"{split}.parquet")
    if not os.path.exists(path):
        print(f"Downloading {split} split...")
        urllib.request.urlretrieve(HF_URL.format(split=split), path)
    table = pq.read_table(path, columns=["image", "labels"]).to_pylist()
    for row in table:
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        img = img.resize((args.size, args.size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=args.quality)
        chunks.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
        offsets.append(offsets[-1] + len(chunks[-1]))
        labels.append(row["labels"])
        split_ids.append(split_id)
    print(f"{split}: {len(table)} images")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"beans{args.size}.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, jpeg=np.concatenate(chunks), offsets=np.array(offsets, dtype=np.int64),
         labels=np.array(labels, dtype=np.int64), split=np.array(split_ids, dtype=np.int64))
print(f"Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
