"""One-time: train the autoencoder notebook's models (bottleneck 2, 4 and 48) for longer and save them.

Usage (from session2_inside_the_black_box/src):
    python tools/pretrain_ae.py [--epochs 40]

Output: data/ae_chachapoyas.pt containing {"columns", "models": {latent: {"state_dict", "history"}}}.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ae_common as ac  # noqa: E402
import rnn_common as rc  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
args = parser.parse_args()

days = ac.SensorDays(rc.load_weather(rc.fetch(rc.WEATHER_FILE)))
x_train = days.scale(days.sliding("train"))
x_val = days.scale(days.sliding("validation", stride=24))

models = {}
for latent in (2, 4, 48):
    torch.manual_seed(0)
    model = ac.SensorAE(latent, len(days.columns))
    history = ac.train_ae(model, x_train, x_val, epochs=args.epochs)
    print(f"Bottleneck {latent}: validation loss {history['val_loss'][-1]:.4f} ({history['seconds']:.0f} s)")
    models[latent] = {"state_dict": model.state_dict(), "history": history}

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ac.WEIGHTS_FILE)
torch.save({"columns": days.columns, "models": models}, out)
print(f"Saved {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
