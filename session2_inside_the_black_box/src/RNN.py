import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import urllib.request
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # Shared model/data code lives in rnn_common.py next to this notebook; download it if missing (e.g. molab)
    try:
        import rnn_common as rc
    except ImportError:
        _cache = os.path.join(os.path.expanduser("~"), ".cache", "untrm_workshop")
        os.makedirs(_cache, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
            "session2_inside_the_black_box/src/rnn_common.py",
            os.path.join(_cache, "rnn_common.py"),
        )
        sys.path.insert(0, _cache)
        import rnn_common as rc

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return mo, np, plt, rc, torch


@app.cell(hide_code=True)
def _(np, plt, rc, torch):
    # HELPER FUNCTIONS (plotting)
    from matplotlib.patches import FancyBboxPatch

    # Projection-friendly defaults: large fonts, thick lines
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "lines.linewidth": 3, "lines.markersize": 9,
    })
    # Okabe-Ito colorblind-safe palette
    BLUE, ORANGE, SKY, VERMILLION, GREEN, PURPLE = "#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"

    def label(name):
        _, text, unit = rc.VARIABLES[name]
        return f"{text} ({unit})"

    def plot_memory_demo(times, rain, memory, r):
        """Rainfall (input) and the one-neuron hidden state ('soil moisture') over time."""
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
        axes[0].bar(times, rain, width=0.04, color=SKY)
        axes[0].set_ylabel("Rain (mm/h)")
        axes[0].set_title("Input: hourly rainfall")
        axes[1].plot(times, memory, color=VERMILLION)
        axes[1].set_ylabel("Hidden state")
        axes[1].set_title(f"Hidden state ('soil moisture') with memory r = {r:.2f}")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def plot_unrolled_diagram(n=5):
        """The same RNN cell drawn once per hour, passing its hidden state to the right."""
        fig, ax = plt.subplots(figsize=(13, 3.6))
        for t in range(n):
            x0 = t * 2.6
            ax.add_patch(FancyBboxPatch((x0, 1.0), 1.6, 1.1, boxstyle="round,pad=0.08",
                                        facecolor=GREEN, edgecolor="black", linewidth=1.5))
            ax.text(x0 + 0.8, 1.55, "RNN\ncell", ha="center", va="center", color="white", fontsize=14,
                    fontweight="bold")
            ax.annotate("", xy=(x0 + 0.8, 0.95), xytext=(x0 + 0.8, 0.25),
                        arrowprops=dict(arrowstyle="-|>", lw=2.5, color=SKY))
            ax.text(x0 + 0.8, 0.0, f"input hour {t + 1}", ha="center", va="center", fontsize=13)
            if t < n - 1:
                ax.annotate("", xy=(x0 + 2.55, 1.55), xytext=(x0 + 1.75, 1.55),
                            arrowprops=dict(arrowstyle="-|>", lw=2.5, color=VERMILLION))
                ax.text(x0 + 2.15, 1.85, f"h{t + 1}", ha="center", fontsize=13, color=VERMILLION)
        ax.annotate("", xy=(n * 2.6 - 0.6, 2.75), xytext=(n * 2.6 - 1.8, 2.15),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black"))
        ax.text(n * 2.6 - 0.5, 2.85, "forecast", va="center", fontsize=13)
        ax.set_xlim(-0.4, n * 2.6 + 0.8)
        ax.set_ylim(-0.3, 3.2)
        ax.axis("off")
        ax.set_title("Unrolled in time: the same cell (same weights) is reused every hour")
        fig.tight_layout()
        return fig

    def plot_weather(df, start, hours=14 * 24):
        """Temperature and humidity for a two-week period."""
        part = df.iloc[start:start + hours]
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        axes[0].plot(part["time"], part["temperature"], color=VERMILLION)
        axes[0].set_ylabel(label("temperature"))
        axes[1].plot(part["time"], part["humidity"], color=BLUE)
        axes[1].set_ylabel(label("humidity"))
        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[0].set_title("Chachapoyas, hourly (NASA POWER)")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def plot_window_diagram(df, target, now, history, horizon):
        """One forecasting example on the real series: past `history` hours -> value `horizon` hours later."""
        lo, hi = now - history - 6, now + horizon + 12
        part = df.iloc[lo:hi + 1]
        hours = np.arange(lo, hi + 1) - now
        fig, ax = plt.subplots(figsize=(13, 4.2))
        ax.axvspan(-history + 1, 0, color=SKY, alpha=0.25, label=f"Past {history} hours (input)")
        ax.plot(hours, part[target], color="gray", linewidth=2)
        ax.axvline(0, color="black", linestyle=":", linewidth=2)
        ax.text(0.5, ax.get_ylim()[1], " now", va="top", fontsize=14)
        ax.plot([horizon], [df[target].iloc[now + horizon]], "*", color=VERMILLION, markersize=22,
                label=f"Target: {horizon} h ahead")
        ax.set_xlabel("Hours relative to now")
        ax.set_ylabel(label(target))
        ax.legend(loc="lower left", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_mae_bars(maes, unit, title):
        """Horizontal bars of test MAE for baselines (and the model, if given)."""
        colors = {"Persistence": PURPLE, "Same hour yesterday": ORANGE}
        names = list(maes)
        fig, ax = plt.subplots(figsize=(10, 0.9 * len(names) + 1.3))
        bars = ax.barh(names, [maes[n] for n in names], color=[colors.get(n, GREEN) for n in names])
        for bar, n in zip(bars, names):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {maes[n]:.2f} {unit}", va="center",
                    fontsize=15)
        ax.invert_yaxis()
        ax.set_xlim(0, max(maes.values()) * 1.3)
        ax.set_xlabel(f"Mean absolute error on the test period ({unit})")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def plot_training(history, yesterday_val, unit):
        """Training and validation MAE per epoch, with the same-hour-yesterday baseline on validation."""
        epochs = np.arange(1, len(history["train_mae"]) + 1)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(epochs, history["train_mae"], "o-", color=BLUE, label="Training")
        ax.plot(epochs, history["val_mae"], "s--", color=GREEN, label="Validation")
        ax.axhline(yesterday_val, color=ORANGE, linestyle=":", label="Same hour yesterday (validation)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Mean absolute error ({unit})")
        ax.set_title("Learning curve")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_forecast(df, fc, windows, i, pred):
        """One test window: history, what really happened, and the forecasts at +F hours."""
        now, F, target = int(windows["now"][i]), fc.horizon, fc.target
        lo = now - fc.history + 1
        hours = np.arange(lo, now + F + 1) - now
        series = df[target].iloc[lo:now + F + 1].values
        fig, ax = plt.subplots(figsize=(13, 4.8))
        ax.plot(hours[hours <= 0], series[hours <= 0], color="gray", label="History (input)")
        ax.plot(hours[hours >= 0], series[hours >= 0], color="gray", linestyle="--", linewidth=2,
                label="What really happened")
        # Forecast markers are spread slightly around +F so they do not hide each other
        dx = max(0.6, (F + fc.history) / 60)
        ax.plot([F], [windows["y"][i]], "o", color="black", markersize=13, label="True value")
        ax.plot([F - dx], [pred[i]], "*", color=GREEN, markersize=22, label="LSTM")
        ax.plot([F + dx], [windows["persistence"][i]], "s", color=PURPLE, markersize=11, label="Persistence")
        ax.plot([F + 2 * dx], [windows["yesterday"][i]], "D", color=ORANGE, markersize=11,
                label="Same hour yesterday")
        ax.axvline(0, color="black", linestyle=":", linewidth=2)
        ax.set_xlabel("Hours relative to now")
        ax.set_ylabel(label(target))
        ax.set_title(f"Forecast for {df['time'].iloc[now + F]:%Y-%m-%d %H:%M} ({F} h ahead)")
        ax.legend(loc="upper left", fontsize=11, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_hidden_units(model, fc, df, start, hours=72, n_units=4):
        """Run the network over `hours` hours and plot the hidden units that best follow the target."""
        x = ((df[fc.inputs].values[start:start + hours] - fc.mean) / fc.std).astype(np.float32)
        with torch.no_grad():
            states = model.hidden_states(torch.from_numpy(x)[None])[0].numpy()  # (hours, hidden)
        target = df[fc.target].values[start:start + hours]
        corr = np.nan_to_num([abs(np.corrcoef(states[:, k], target)[0, 1]) for k in range(states.shape[1])])
        best = np.argsort(corr)[::-1][:n_units]
        times = df["time"].iloc[start:start + hours]
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        axes[0].plot(times, target, color=VERMILLION)
        axes[0].set_ylabel(label(fc.target))
        axes[0].set_title("Input series")
        for k, color in zip(best, [BLUE, GREEN, PURPLE, ORANGE]):
            axes[1].plot(times, states[:, k], color=color, linewidth=2.5, label=f"Unit {k + 1}")
        axes[1].set_ylabel("Hidden state")
        axes[1].set_title(f"{n_units} of the {states.shape[1]} hidden units, read hour by hour")
        axes[1].legend(loc="upper left", fontsize=12, ncol=n_units)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def plot_lstm_gates():
        """Conceptual LSTM: a memory line with forget, input and output gates."""
        fig, ax = plt.subplots(figsize=(12, 3.4))
        ax.annotate("", xy=(11.2, 2.4), xytext=(0.2, 2.4), arrowprops=dict(arrowstyle="-|>", lw=4, color=BLUE))
        ax.text(0.2, 2.75, "Long-term memory (cell state)", fontsize=14, color=BLUE)
        for x, name, text, color in [(2.5, "Forget gate", "what to\nerase", VERMILLION),
                                     (5.5, "Input gate", "what to\nadd", GREEN),
                                     (8.5, "Output gate", "what to\nreport", ORANGE)]:
            ax.add_patch(FancyBboxPatch((x - 0.9, 0.5), 1.8, 1.1, boxstyle="round,pad=0.08", facecolor=color,
                                        edgecolor="black"))
            ax.text(x, 1.05, text, ha="center", va="center", fontsize=13, color="white", fontweight="bold")
            ax.text(x, 0.15, name, ha="center", va="center", fontsize=14)
            ax.annotate("", xy=(x, 2.3), xytext=(x, 1.7), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=color))
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.2, 3.1)
        ax.axis("off")
        fig.tight_layout()
        return fig

    def plot_time_encoding():
        """Hour as one number (jumps at midnight) vs. sin/cos encoding (smooth), and the clock circle."""
        hours = np.arange(0, 48.01, 0.25)
        hod = hours % 24
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 3, 2.2]})
        # 1) Single number: a jump of 23 at every midnight
        axes[0].plot(hours, np.where(np.diff(hod, prepend=0) < 0, np.nan, hod), color=PURPLE)
        axes[0].annotate("jump from 23\nback to 0", xy=(24, 12), xytext=(27.5, 15), fontsize=13,
                         arrowprops=dict(arrowstyle="-|>", lw=2))
        axes[0].set_title("One number: hour (0–23)")
        axes[0].set_ylabel("Value given to the network")
        # 2) Two numbers: smooth through midnight
        axes[1].plot(hours, np.sin(2 * np.pi * hod / 24), color=BLUE, label="sin(2π·hour/24)")
        axes[1].plot(hours, np.cos(2 * np.pi * hod / 24), color=ORANGE, label="cos(2π·hour/24)")
        axes[1].set_title("Two numbers: sine and cosine")
        axes[1].legend(loc="upper center", fontsize=12, ncol=2, framealpha=0.95)
        axes[1].set_ylim(-1.25, 1.75)
        for ax in axes[:2]:
            for midnight in (24, 48):
                ax.axvline(midnight, color="gray", linestyle=":", linewidth=2)
            ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48], ["0", "6", "12", "18", "0", "6", "12", "18", "0"])
            ax.set_xlabel("Hour of day (two days)")
            ax.grid(True, alpha=0.3)
        # 3) Clock circle: every hour is a point (cos, sin); 23:00 sits next to 00:00
        h = np.arange(24)
        x, y = np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24)
        axes[2].plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)),
                     color="#CCCCCC", linewidth=2)
        axes[2].scatter(x, y, s=60, color=BLUE, zorder=3)
        for k, (dx, dy, ha) in {0: (0.15, 0.2, "left"), 23: (0.15, -0.3, "left"), 6: (0, 0.3, "center"),
                                12: (-0.15, 0, "right"), 18: (0, -0.3, "center")}.items():
            axes[2].text(x[k] + dx, y[k] + dy, f"{k}:00", ha=ha, va="center", fontsize=13,
                         color=VERMILLION if k in (0, 23) else "black")
        axes[2].plot(x[[23, 0]], y[[23, 0]], color=VERMILLION, linewidth=4)
        axes[2].set_title("On a circle")
        axes[2].set_xlabel("cos")
        axes[2].set_ylabel("sin")
        axes[2].set_aspect("equal")
        axes[2].set_xlim(-2.0, 2.2)
        axes[2].set_ylim(-1.6, 1.6)
        axes[2].set_xticks([-1, 0, 1])
        axes[2].set_yticks([-1, 0, 1])
        fig.tight_layout()
        return fig

    def plot_lstm_unrolled(n_inputs, hidden, history, horizon, target_name):
        """The network we train, unrolled over the input hours: one LSTM layer, then a linear layer on the last step."""
        fig, ax = plt.subplots(figsize=(14, 5.2))
        steps = [("hour 1", 0.0), ("hour 2", 2.6), (f"hour {history - 1}", 7.0), (f"hour {history}", 9.6)]
        y_cell, w, hgt = 1.5, 1.7, 1.2
        for k, (name, x0) in enumerate(steps):
            last = k == len(steps) - 1
            ax.add_patch(FancyBboxPatch((x0, y_cell), w, hgt, boxstyle="round,pad=0.08", facecolor=GREEN,
                                        edgecolor="black", linewidth=1.5))
            ax.text(x0 + w / 2, y_cell + hgt / 2, f"LSTM\n{hidden} units", ha="center", va="center", color="white",
                    fontsize=13, fontweight="bold")
            # Input at this hour: one vector with all input columns
            ax.annotate("", xy=(x0 + w / 2, y_cell - 0.1), xytext=(x0 + w / 2, 0.55),
                        arrowprops=dict(arrowstyle="-|>", lw=2.5, color=SKY))
            ax.text(x0 + w / 2, 0.25, f"{name}\n({n_inputs} values)", ha="center", va="center", fontsize=12)
            # Output at this hour: only the last one is used
            if not last:
                ax.annotate("", xy=(x0 + w / 2, y_cell + hgt + 0.55), xytext=(x0 + w / 2, y_cell + hgt + 0.1),
                            arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#BBBBBB"))
                ax.text(x0 + w / 2, y_cell + hgt + 0.75, "not used", ha="center", fontsize=11, color="#999999")
            # Memory (c) and hidden state (h) passed to the next hour
            nxt = steps[k + 1][1] if not last else None
            if nxt is not None:
                gap_text = k == 1  # between hour 2 and hour H-1 the hours are skipped
                for dy, color, lab in [(0.85, BLUE, "c"), (0.35, VERMILLION, "h")]:
                    ax.annotate("", xy=(nxt - 0.05, y_cell + dy), xytext=(x0 + w + 0.05, y_cell + dy),
                                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=color,
                                                linestyle="--" if gap_text else "-"))
                    ax.text((x0 + w + nxt) / 2, y_cell + dy + 0.08, lab, ha="center", fontsize=13, color=color)
                if gap_text:
                    ax.text((x0 + w + nxt) / 2, y_cell - 0.35, "· · ·", ha="center", fontsize=22)
        # Final hidden state -> linear layer -> forecast
        x_last = steps[-1][1] + w
        ax.annotate("", xy=(x_last + 0.9, y_cell + hgt / 2), xytext=(x_last + 0.05, y_cell + hgt / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=VERMILLION))
        ax.text(x_last + 0.47, y_cell + hgt / 2 + 0.12, f"h ({hidden})", ha="center", fontsize=12, color=VERMILLION)
        ax.add_patch(FancyBboxPatch((x_last + 0.95, y_cell + 0.15), 1.3, 0.9, boxstyle="round,pad=0.08",
                                    facecolor=ORANGE, edgecolor="black", linewidth=1.5))
        ax.text(x_last + 1.6, y_cell + 0.6, "Linear", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.annotate("", xy=(x_last + 1.6, y_cell + hgt + 0.75), xytext=(x_last + 1.6, y_cell + 1.15),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black"))
        ax.text(x_last + 1.6, y_cell + hgt + 0.95, f"{target_name}\n{horizon} h ahead", ha="center", va="bottom",
                fontsize=13)
        ax.text(0.0, 4.7, "Same LSTM weights at every hour", fontsize=14, color=GREEN, fontweight="bold")
        ax.text(0.0, 4.3, "c = long-term memory,  h = hidden state", fontsize=13)
        ax.set_xlim(-0.3, x_last + 2.8)
        ax.set_ylim(-0.2, 5.0)
        ax.axis("off")
        fig.tight_layout()
        return fig

    return (
        plot_forecast,
        plot_hidden_units,
        plot_lstm_gates,
        plot_lstm_unrolled,
        plot_mae_bars,
        plot_memory_demo,
        plot_time_encoding,
        plot_training,
        plot_unrolled_diagram,
        plot_weather,
        plot_window_diagram,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        # Recurrent Networks: Forecasting the Weather in Chachapoyas

        **Step 1:** memory by hand · **Step 2:** data and task · **Step 3:** baselines · **Step 4:** train an
        LSTM · **Step 5:** predictions
        """),
        mo.accordion({
            "More details: where this fits": mo.md(r"""
            A CNN re-uses the same small filter at every position of an **image**. A recurrent neural network
            (**RNN**) re-uses the same small network at every step of a **sequence**: hour after hour of
            weather, day after day of sensor readings, or letter after letter of DNA. While it reads, it keeps
            a running summary of what it has seen so far, called the **hidden state**. In this notebook we
            will build intuition for the hidden state, then train an RNN (an **LSTM**) to forecast the
            temperature in Chachapoyas, and always compare it with simple baselines.
            """)
        }),
    ])
    return


@app.cell
def _(mo, rc):
    # Load hourly weather for Chachapoyas (2023-2024), from the local data/ folder or GitHub
    try:
        weather = rc.load_weather(rc.fetch(rc.WEATHER_FILE))
        weather_error = None
    except Exception as _e:
        weather, weather_error = None, str(_e)
    mo.stop(
        weather is None,
        mo.callout(mo.md(f"**Could not load the weather data.** Check the internet connection.\n\n`{weather_error}`"),
                   kind="danger"),
    )
    return (weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 1: Memory by hand

        An RNN keeps a running memory, its **hidden state**, updated every hour from the new input and the
        previous state.
        """),
        mo.accordion({
            "More details: the hidden state": mo.md(r"""
            Think of **soil moisture**. It goes up when it rains and slowly dries out afterwards, so it
            "remembers" past rain. We can write this as a tiny RNN with a single neuron:

            *new state = r × previous state + rain this hour*

            The number **r** controls the memory: with r = 0 the state only reflects the current hour; with
            r close to 1 it remembers rain from days ago. The **same rule, with the same numbers, is applied
            every hour**: this is **weight sharing across time**, just as a CNN shares its filter across
            space.

            A real RNN works the same way, but it has many hidden units instead of one, adds a non-linear
            activation, and **learns** its numbers from data instead of us choosing them.
            """)
        }),
    ])
    return


@app.cell
def _(mo):
    memory_slider = mo.ui.slider(0.0, 0.99, step=0.01, value=0.9, label="Memory r")
    memory_slider
    return (memory_slider,)


@app.cell
def _(memory_slider, np, plot_memory_demo, weather):
    # Three rainiest weeks of the record, and the one-neuron RNN applied hour by hour
    _rain = weather["rain"].values
    _weekly = np.convolve(_rain, np.ones(21 * 24), mode="valid")
    _start = int(np.argmax(_weekly))
    _x = _rain[_start:_start + 21 * 24]

    _h = np.zeros_like(_x)
    for _t in range(len(_x)):
        _h[_t] = memory_slider.value * (_h[_t - 1] if _t > 0 else 0.0) + _x[_t]

    plot_memory_demo(weather["time"].iloc[_start:_start + 21 * 24], _x, _h, memory_slider.value)
    return


@app.cell
def _(plot_unrolled_diagram):
    plot_unrolled_diagram()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Try this": mo.callout(
            mo.md(r"""
            Move the memory slider to **0**: the hidden state is just the rain of the current hour, with no
            memory. Move it to **0.99**: each rain event lifts the state for days, like wet soil after a
            storm. An RNN learns, from data, how much of the past is worth keeping.
            """),
            kind="info",
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 2: The data and the forecasting task

        Hourly weather for Chachapoyas (2023–2024). Task: predict a value **F hours ahead** from the **last
        H hours**.
        """),
        mo.accordion({
            "More details: where the data comes from": mo.md(r"""
            The data comes from **NASA POWER**, a free service that estimates hourly weather for any point
            on Earth from satellites and weather models. It is **not** a weather station: each value
            represents an area of roughly 50 km around Chachapoyas, so local effects (valleys, slopes) are
            smoothed out. The notebook can use a real station series instead (a CSV with a time column
            and columns such as temperature, humidity and rainfall).
            """),
            "More details: training, validation and test periods": mo.md(r"""
            With time series we **never shuffle** before splitting: the model must be tested on a period
            that comes **after** everything it learned from, as in real forecasting. We use the first 70%
            of the record (January 2023 to late May 2024) for training, the next 15% (June to mid-September
            2024) for validation, and the last 15% (mid-September to December 2024) as the test period. Each variable is rescaled using
            the mean and spread of the training period only.
            """),
        }),
    ])
    return


@app.cell
def _(mo):
    week_slider = mo.ui.slider(0, 102, step=1, value=30, label="Week to show")
    week_slider
    return (week_slider,)


@app.cell
def _(plot_weather, weather, week_slider):
    plot_weather(weather, week_slider.value * 7 * 24)
    return


@app.cell
def _(mo, rc):
    target_dropdown = mo.ui.dropdown(
        options={"Temperature (°C)": "temperature", "Relative humidity (%)": "humidity"},
        value="Temperature (°C)", label="Predict",
    )
    history_slider = mo.ui.slider(24, 120, step=12, value=48, label="History H (hours)", debounce=True)
    horizon_slider = mo.ui.slider(1, 24, step=1, value=6, label="Horizon F (hours)", debounce=True)
    inputs_select = mo.ui.multiselect(
        options={rc.VARIABLES[k][1]: k for k in rc.VARIABLES},
        value=["Temperature", "Relative humidity", "Rainfall"], label="Input variables",
    )
    mo.vstack([
        mo.hstack([target_dropdown, history_slider, horizon_slider], justify="start", gap=2),
        inputs_select,
    ])
    return history_slider, horizon_slider, inputs_select, target_dropdown


@app.cell
def _(
    history_slider,
    horizon_slider,
    inputs_select,
    mo,
    rc,
    target_dropdown,
    weather,
):
    mo.stop(not inputs_select.value, mo.callout(mo.md("Select at least one input variable."), kind="warn"))
    # Build the forecasting windows for the chosen settings (time of day is always added as an input)
    forecaster = rc.Forecaster(weather, target_dropdown.value, inputs_select.value,
                               history_slider.value, horizon_slider.value)
    windows = {name: forecaster.windows(name) for name in ["train", "validation", "test"]}
    return forecaster, windows


@app.cell
def _(
    forecaster,
    mo,
    plot_time_encoding,
    plot_window_diagram,
    rc,
    weather,
    windows,
):
    mo.vstack([
        plot_window_diagram(weather, forecaster.target, int(windows["test"]["now"][300]),
                            forecaster.history, forecaster.horizon),
        mo.md(
            f"Every hour of the record gives one example like this: **{len(windows['train']['y']):,}** training, "
            f"**{len(windows['validation']['y']):,}** validation and **{len(windows['test']['y']):,}** test "
            f"examples. Each input is a table of **{forecaster.history} hours × {len(forecaster.inputs)} columns**: "
            f"the {len(forecaster.inputs) - len(rc.HOUR_FEATURES)} selected variable(s) plus "
            f"{len(rc.HOUR_FEATURES)} columns for the time of day."
        ),
        mo.accordion({
            "More details: why does the time of day use two columns?": mo.vstack([
                mo.md(r"""
                If we gave the network the hour as a single number (0 to 23), 23:00 and 00:00 would look very
                far apart (23 vs. 0), although they are only one hour apart: the value **jumps** at every
                midnight (left). Instead we place the hour on a circle, like the hand of a clock, and give the
                network its two coordinates: **sin(2π·hour/24)** and **cos(2π·hour/24)** (middle). Both change
                smoothly through midnight, and on the circle 23:00 sits right next to 00:00 (right). Together
                the two numbers still identify every hour uniquely, so the network can easily tell morning from
                afternoon.
                """),
                plot_time_encoding(),
                mo.md(r"""
                *Looking ahead:* **transformers** (Part 5) use the same trick to tell the network **where** each
                element of a sequence is. Their **positional encoding** gives every position several sine and
                cosine pairs, each turning at a different speed, like the hour, minute and second hands of a
                clock.
                """),
            ])
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 3: Baselines first

        Before any neural network: how good are the simplest possible guesses?
        """),
        mo.accordion({
            "More details: the two baselines": mo.md(r"""
            - **Persistence:** "in F hours it will be the same as now".
            - **Same hour yesterday:** "in F hours it will be the same as at that hour yesterday". This uses
              the strong daily cycle of temperature and humidity in Chachapoyas.

            A model is only useful if it beats these free guesses on the test period. The error is the
            **mean absolute error (MAE)**: the average size of the mistake, in °C or %.
            """)
        }),
    ])
    return


@app.cell
def _(forecaster, plot_mae_bars, rc, windows):
    _test = windows["test"]
    baseline_maes = {
        "Persistence": rc.mae(_test["persistence"], _test["y"]),
        "Same hour yesterday": rc.mae(_test["yesterday"], _test["y"]),
    }
    plot_mae_bars(baseline_maes, rc.VARIABLES[forecaster.target][2], f"Baselines, {forecaster.horizon} h ahead")
    return


@app.cell(hide_code=True)
def _(mo, plot_lstm_gates):
    mo.vstack([
        mo.md(r"""
        ## Step 4: Train an LSTM

        An **LSTM** is an RNN with **gates** that decide what to keep, what to add and what to report.
        """),
        mo.accordion({
            "More details: what does an LSTM add?": mo.vstack([
                mo.md(r"""
                A plain RNN tends to forget: information from many hours ago fades away as it is mixed into
                the hidden state again and again. An LSTM (Long Short-Term Memory) adds a separate
                **long-term memory** and three learned **gates**:

                - **Forget gate:** what to erase from memory (for example, yesterday's rain once the soil dried).
                - **Input gate:** what new information to store.
                - **Output gate:** what part of the memory to use for the current output.

                The gates give the network a *choice* about what to remember, instead of forgetting by
                default.
                """),
                plot_lstm_gates(),
            ])
        }),
    ])
    return


@app.cell
def _(mo):
    hidden_dropdown = mo.ui.dropdown(options=["16", "32", "64"], value="32", label="Hidden units")
    epochs_slider = mo.ui.slider(5, 30, step=1, value=15, label="Epochs", debounce=True)
    mo.hstack([hidden_dropdown, epochs_slider], justify="start", gap=2)
    return epochs_slider, hidden_dropdown


@app.cell
def _(epochs_slider, forecaster, hidden_dropdown, mo, windows):
    # Rough CPU cost per training example per input hour, measured on 2 CPU threads
    _sec = {"16": 0.6e-6, "32": 1.1e-6, "64": 2.5e-6}[hidden_dropdown.value]
    _estimate = _sec * len(windows["train"]["y"]) * forecaster.history * epochs_slider.value
    mo.md(f"*Estimated training time: about {max(1, round(_estimate))} s*"
          + (" (over a minute: consider fewer epochs or a shorter history)" if _estimate > 60 else ""))
    return


@app.cell
def _(forecaster, hidden_dropdown, mo, plot_lstm_unrolled, rc):
    mo.vstack([
        mo.md("**The network we train, unrolled over the input hours** (updates with the settings above and in Step 2):"),
        plot_lstm_unrolled(len(forecaster.inputs), int(hidden_dropdown.value), forecaster.history,
                           forecaster.horizon, rc.VARIABLES[forecaster.target][1]),
        mo.accordion({
            "More details: reading the diagram": mo.md(rf"""
            There is **one LSTM layer** with {hidden_dropdown.value} hidden units. It is drawn once per hour
            only to show time passing; all the boxes are the **same layer with the same weights**. At every
            hour it reads that hour's {len(forecaster.inputs)} input values and passes two things to the next
            hour: its long-term memory (**c**) and its hidden state (**h**).

            The LSTM also produces an output at every hour, but we **only use the one after the last hour**
            ({forecaster.history}): by then the hidden state summarizes the whole window. A single
            **linear layer** turns those {hidden_dropdown.value} numbers into one forecast,
            {forecaster.horizon} hours ahead.
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        **Train Model** trains a new LSTM with the settings above; **Load Pre-trained** loads one trained
        longer beforehand (shown when the notebook opens).
        """),
        mo.accordion({
            "More details: live training vs. pre-trained": mo.md(r"""
            **Train Model** starts from random weights and uses the target, history, horizon and input
            variables chosen in Step 2. **Load Pre-trained** loads an LSTM trained beforehand for 40 epochs
            with fixed settings: temperature, 48 hours of history, 6 hours ahead, inputs temperature,
            humidity and rainfall, 32 hidden units. When it is loaded, the results below use those settings.
            """)
        }),
    ])
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train Model")
    pretrained_button = mo.ui.run_button(label="Load Pre-trained", kind="neutral")
    mo.hstack([train_button, pretrained_button], justify="start")
    return pretrained_button, train_button


@app.cell
def _(
    epochs_slider,
    forecaster,
    hidden_dropdown,
    mo,
    pretrained_button,
    rc,
    torch,
    train_button,
    weather,
    windows,
):

    # Pre-trained by default (when the notebook opens or Load Pre-trained is pressed); Train Model trains live
    pretrained_button  # pressing it re-runs this cell
    if not train_button.value:
        _ckpt = torch.load(rc.fetch(rc.WEIGHTS_FILE), map_location="cpu", weights_only=False)
        _cfg = _ckpt["config"]
        rnn_fc = rc.Forecaster(weather, _cfg["target"], _cfg["inputs"], _cfg["history"], _cfg["horizon"],
                               _cfg["use_hour"])
        rnn_windows = {name: rnn_fc.windows(name) for name in ["validation", "test"]}
        rnn_model = rc.WeatherRNN(len(rnn_fc.inputs), _cfg["hidden"], _cfg["cell"])
        rnn_model.load_state_dict(_ckpt["state_dict"])
        rnn_history = _ckpt["history"]
        rnn_source = (f"pre-trained LSTM ({len(rnn_history['train_mae'])} epochs; {_cfg['target']}, "
                      f"{_cfg['history']} h history, {_cfg['horizon']} h ahead)")
    else:
        torch.manual_seed(0)
        rnn_fc, rnn_windows = forecaster, windows
        rnn_model = rc.WeatherRNN(len(forecaster.inputs), int(hidden_dropdown.value))
        with mo.status.spinner(title="Training the LSTM..."):
            rnn_history = rc.train_model(rnn_model, forecaster, windows["train"], windows["validation"],
                                         epochs=epochs_slider.value)
        rnn_source = f"LSTM trained live ({epochs_slider.value} epochs, {rnn_history['seconds']:.0f} s)"
    return rnn_fc, rnn_history, rnn_model, rnn_source, rnn_windows


@app.cell
def _(
    mo,
    plot_mae_bars,
    plot_training,
    rc,
    rnn_fc,
    rnn_history,
    rnn_model,
    rnn_source,
    rnn_windows,
):
    _unit = rc.VARIABLES[rnn_fc.target][2]
    _test, _val = rnn_windows["test"], rnn_windows["validation"]
    test_pred = rc.predict(rnn_model, _test["X"], rnn_fc)
    _maes = {
        "Persistence": rc.mae(_test["persistence"], _test["y"]),
        "Same hour yesterday": rc.mae(_test["yesterday"], _test["y"]),
        "LSTM": rc.mae(test_pred, _test["y"]),
    }
    mo.vstack([
        mo.md(f"**Results for the {rnn_source}**"),
        plot_training(rnn_history, rc.mae(_val["yesterday"], _val["y"]), _unit),
        plot_mae_bars(_maes, _unit, f"Test period, {rnn_fc.horizon} h ahead"),
    ])
    return (test_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Learn more: reading the results": mo.md(r"""
        - The learning curve shows the average error on the training and validation periods after each epoch.
          The dotted line is the "same hour yesterday" baseline on the validation period: the LSTM is only
          useful once its validation curve drops below it.
        - The bar chart compares all three on the **test period**, which was never used during training.
        - Try a horizon of 1 hour and of 24 hours. When does the LSTM help most?
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 5: Predictions and where they fail

        History, what really happened, and each forecast for one test example.
        """),
        mo.accordion({
            "More details: reading the forecast plot": mo.md(r"""
            The gray line on the left is the input the network read. The dashed line is what really happened
            afterwards; the black dot is the true value F hours ahead. The green star is the LSTM's forecast,
            and the square and diamond are the two baselines. Turn on **Show the worst prediction** to see the
            test example with the largest LSTM error: these are often sudden changes, such as a cold rainy
            afternoon, that the past hours did not announce.
            """)
        }),
    ])
    return


@app.cell
def _(mo, rnn_windows):
    example_slider = mo.ui.slider(0, len(rnn_windows["test"]["y"]) - 1, step=1, value=300, label="Test example")
    worst_switch = mo.ui.switch(label="Show the worst prediction")
    mo.hstack([example_slider, worst_switch], justify="start", gap=2)
    return example_slider, worst_switch


@app.cell
def _(
    example_slider,
    np,
    plot_forecast,
    rnn_fc,
    rnn_windows,
    test_pred,
    weather,
    worst_switch,
):
    _test = rnn_windows["test"]
    selected_example = int(np.argmax(np.abs(test_pred - _test["y"]))) if worst_switch.value else example_slider.value
    plot_forecast(weather, rnn_fc, _test, selected_example, test_pred)
    return (selected_example,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### Look inside: the hidden state

        Some hidden units of the trained LSTM, read hour by hour over three days.
        """),
        mo.accordion({
            "More details: what the hidden units track": mo.md(r"""
            We feed three days of weather into the trained network and record its hidden state after every
            hour. We show the units whose values follow the input series most closely. Many of them rise and
            fall with the **daily cycle**, even though nobody told the network that days exist: it learned
            that this rhythm is useful for forecasting. This is the RNN version of the learned filters we saw
            in the CNN.
            """)
        }),
    ])
    return


@app.cell
def _(
    plot_hidden_units,
    rnn_fc,
    rnn_model,
    rnn_windows,
    selected_example,
    weather,
):
    plot_hidden_units(rnn_model, rnn_fc, weather, int(rnn_windows["test"]["start"][selected_example]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            **Where this is useful:** temperature and humidity forecasts for **coffee leaf rust** risk, soil
            moisture for **irrigation**, and any stream of **sensor** readings in the field.
            """),
            kind="info",
        ),
        mo.accordion({
            "More details: from forecasts to decisions": mo.md(r"""
            Coffee leaf rust spreads when leaves stay wet and temperatures are mild, so a forecast of humid
            hours a day ahead can tell a farmer when to inspect or protect plants (the orchestration example
            from Session 1). The same window → forecast recipe works for soil moisture, river levels or
            greenhouse sensors: only the input columns change.
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to notice

    - An RNN reads a sequence **one step at a time** and carries a **hidden state** forward; the same weights
      are reused at every step.
    - An **LSTM** adds gates that decide what to remember, so it can use information from many hours back.
    - A forecast is only useful if it **beats simple baselines** such as persistence and "same hour yesterday".

    ## Questions for discussion

    1. Does a longer history always help? Try 24 hours and 120 hours.
    2. When does the LSTM beat the baselines by the most: 1 hour ahead or 24 hours ahead? Why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "References": mo.md(r"""
        - **Recurrent networks:** Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2),
          179–211. [doi:10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
        - **Training an unrolled network:** Werbos, P. J. (1990). Backpropagation through time: what it does and
          how to do it. *Proceedings of the IEEE*, 78(10), 1550–1560.
          [doi:10.1109/5.58337](https://doi.org/10.1109/5.58337)
        - **LSTM:** Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*,
          9(8), 1735–1780. [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
        - **The forget gate:** Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual
          prediction with LSTM. *Neural Computation*, 12(10), 2451–2471.
          [doi:10.1162/089976600300015015](https://doi.org/10.1162/089976600300015015)
        - **Forecasting and simple baselines:** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting:
          principles and practice*, 3rd edition. OTexts: Melbourne, Australia. [otexts.com/fpp3](https://otexts.com/fpp3/)
        - **Positional encoding (looking ahead):** Vaswani, A., et al. (2017). Attention is all you need.
          *Advances in Neural Information Processing Systems 30* (NIPS 2017).
          [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
        - **Adam optimizer:** Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
          *ICLR 2015*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
        - **Weather data:** NASA POWER (Prediction Of Worldwide Energy Resources), hourly API, community AG,
          Chachapoyas (-6.23, -77.87), 2023–2024, accessed November 2026. The data was obtained from the
          National Aeronautics and Space Administration (NASA) Langley Research Center's POWER project funded
          through the NASA Earth Science Division. [power.larc.nasa.gov](https://power.larc.nasa.gov/)
        - **PyTorch** (the library used to build and train the networks): Paszke, A., et al. (2019). PyTorch:
          An imperative style, high-performance deep learning library. *Advances in Neural Information
          Processing Systems 32* (NeurIPS 2019). [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
        """)
    })
    return


if __name__ == "__main__":
    app.run()
