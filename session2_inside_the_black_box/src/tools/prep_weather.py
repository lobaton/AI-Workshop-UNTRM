"""One-time prep: download hourly NASA POWER weather for Chachapoyas and save it as a CSV.

Usage (from session2_inside_the_black_box/src):
    python tools/prep_weather.py [--lat -6.23] [--lon -77.87] [--start 20230101] [--end 20241231]

Output: data/chachapoyas_hourly.csv with columns
    time (local solar time), temperature (°C), humidity (%), rain (mm/h), wind (m/s), pressure (kPa)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rnn_common as rc  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--lat", type=float, default=rc.CHACHAPOYAS[0])
parser.add_argument("--lon", type=float, default=rc.CHACHAPOYAS[1])
parser.add_argument("--start", default="20230101")
parser.add_argument("--end", default="20241231")
args = parser.parse_args()

df = rc.fetch_nasa_power(args.lat, args.lon, args.start, args.end)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rc.WEATHER_FILE)
df.to_csv(out, index=False, float_format="%.2f", date_format="%Y-%m-%d %H:%M")
print(df.describe().round(2))
print(f"Saved {out}: {len(df)} hours, {os.path.getsize(out) / 1e6:.2f} MB")
