"""One-time: train the transformer notebook's DNA models (transformer with and without positional encoding, and
the 1D CNN) for longer and save them.

Usage (from session2_inside_the_black_box/src):
    python tools/pretrain_attention.py [--epochs 40]

Output: data/dna_transformer.pt containing
    {"models": {"transformer", "transformer_no_position", "cnn": {"state_dict", "history", "test_acc"}}}.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attention_common as at  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
args = parser.parse_args()

data = at.make_dataset()
(x, y, _), (x_val, y_val, _), (x_test, y_test, _) = data["train"], data["validation"], data["test"]

models = {}
for name, model, lr in [("transformer", at.TinyTransformer(), 3e-3),
                        ("transformer_no_position", at.TinyTransformer(use_position=False), 3e-3),
                        ("cnn", at.MotifCNN(), 1e-2)]:
    torch.manual_seed(0)
    history = at.train_model(model, x, y, x_val, y_val, epochs=args.epochs, lr=lr)
    test_acc = at.accuracy(model, x_test, y_test)
    print(f"{name}: test accuracy {test_acc:.1%} ({history['seconds']:.0f} s)")
    models[name] = {"state_dict": model.state_dict(), "history": history, "test_acc": test_acc}

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), at.WEIGHTS_FILE)
torch.save({"models": models}, out)
print(f"Saved {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
