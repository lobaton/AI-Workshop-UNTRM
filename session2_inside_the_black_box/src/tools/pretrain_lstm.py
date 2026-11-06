"""One-time: train the RNN notebook's default LSTM for longer and save the weights.

Usage (from session2_inside_the_black_box/src):
    python tools/pretrain_lstm.py [--epochs 40]

Output: data/lstm_chachapoyas.pt containing {"config", "state_dict", "history", "test_mae"}.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rnn_common as rc  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
args = parser.parse_args()

config = {"target": "temperature", "inputs": ["temperature", "humidity", "rain"], "history": 48, "horizon": 6,
          "use_hour": True, "hidden": 32, "cell": "LSTM"}

df = rc.load_weather(rc.fetch(rc.WEATHER_FILE))
fc = rc.Forecaster(df, config["target"], config["inputs"], config["history"], config["horizon"], config["use_hour"])
train, val, test = fc.windows("train"), fc.windows("validation"), fc.windows("test")

torch.manual_seed(0)
model = rc.WeatherRNN(len(fc.inputs), config["hidden"], config["cell"])
history = rc.train_model(model, fc, train, val, epochs=args.epochs, log=print)
test_mae = rc.mae(rc.predict(model, test["X"], fc), test["y"])
print(f"Test MAE: {test_mae:.2f} | persistence {rc.mae(test['persistence'], test['y']):.2f} "
      f"| same hour yesterday {rc.mae(test['yesterday'], test['y']):.2f}")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rc.WEIGHTS_FILE)
torch.save({"config": config, "state_dict": model.state_dict(), "history": history, "test_mae": test_mae}, out)
print(f"Saved {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
