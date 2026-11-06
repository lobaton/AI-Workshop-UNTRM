"""One-time: train the notebook's BeansCNN for longer (full training set, augmentation) and save the weights.

Usage (from session2_inside_the_black_box/src):
    python tools/pretrain_cnn.py [--filters 16] [--epochs 40]

Output: data/beans_cnn128.pt containing {"n_filters", "state_dict", "history", "test_acc"}.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cnn_common as cc  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--filters", type=int, default=16)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=3e-3)
args = parser.parse_args()

data = cc.load_beans(cc.fetch(cc.BEANS_FILE))
x_train, y_train = data["train"]
x_val, y_val = data["validation"]
x_test, y_test = data["test"]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model = cc.BeansCNN(args.filters)
history = cc.train_model(
    model, cc.to_tensor(x_train), torch.from_numpy(y_train), cc.to_tensor(x_val), torch.from_numpy(y_val),
    epochs=args.epochs, lr=args.lr, use_augment=True, device=device, log=print,
)
_, test_acc, _ = cc.evaluate(model, cc.to_tensor(x_test), torch.from_numpy(y_test))
print(f"Test accuracy: {test_acc:.1%}")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cc.WEIGHTS_FILE)
torch.save({"n_filters": args.filters, "state_dict": model.state_dict(), "history": history,
            "test_acc": test_acc}, out)
print(f"Saved {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
