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

    # Shared code lives next to this notebook; download it if missing (e.g. molab)
    try:
        import ae_common as ac
        import rnn_common as rc
    except ImportError:
        _cache = os.path.join(os.path.expanduser("~"), ".cache", "untrm_workshop")
        os.makedirs(_cache, exist_ok=True)
        for _name in ("rnn_common.py", "ae_common.py"):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
                f"session2_inside_the_black_box/src/{_name}",
                os.path.join(_cache, _name),
            )
        sys.path.insert(0, _cache)
        import ae_common as ac
        import rnn_common as rc

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return ac, mo, np, plt, rc, torch


@app.cell(hide_code=True)
def _(np, plt):
    # HELPER FUNCTIONS (plotting)
    from matplotlib.patches import Polygon, FancyBboxPatch

    # Projection-friendly defaults: large fonts, thick lines
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "lines.linewidth": 3, "lines.markersize": 9,
    })
    # Okabe-Ito colorblind-safe palette
    BLUE, ORANGE, SKY, VERMILLION, GREEN, PURPLE = "#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"
    HOURS = np.arange(24)

    def plot_pca_day(day, recon, k, errors):
        """One day and its reconstruction from k numbers | reconstruction error vs. k."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [3, 2]})
        axes[0].plot(HOURS, day, "o-", color="gray", label="Original day")
        axes[0].plot(HOURS, recon, color=VERMILLION, label=f"Rebuilt from {k} number{'s' if k > 1 else ''}")
        axes[0].set_xlabel("Hour of day")
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].legend(fontsize=12)
        axes[0].set_title("One day of temperature")
        ks = np.arange(1, len(errors) + 1)
        axes[1].plot(ks, errors, color=BLUE)
        axes[1].plot([k], [errors[k - 1]], "o", color=VERMILLION, markersize=13)
        axes[1].set_xlabel("Numbers kept (k)")
        axes[1].set_ylabel("Average error (°C)")
        axes[1].set_title("Error over all days")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_pca_diagram(k):
        """PCA drawn as an autoencoder: 24 numbers -> k numbers -> 24 numbers, one matrix each way."""
        fig, ax = plt.subplots(figsize=(13, 3.4))
        wide, narrow = 2.2, 0.35 + 1.4 * k / 12
        boxes = [(0.0, wide, "Input\n24 numbers\n(24 h)", "#DDDDDD"),
                 (6.1, narrow, f"{k} number{'s' if k > 1 else ''}", ORANGE),
                 (12.2, wide, "Output\n24 numbers\n(rebuilt day)", "#DDDDDD")]
        for x, h, text, color in boxes:
            ax.add_patch(FancyBboxPatch((x, -h / 2), 1.6, h, boxstyle="round,pad=0.05", facecolor=color,
                                        edgecolor="black"))
            ax.text(x + 0.8, 0, text, ha="center", va="center", fontsize=13)
        ax.add_patch(Polygon([(1.9, wide / 2), (5.9, narrow / 2), (5.9, -narrow / 2), (1.9, -wide / 2)],
                             facecolor=SKY, edgecolor="black"))
        ax.text(3.9, 0, "Encoder\none matrix", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.add_patch(Polygon([(7.9, narrow / 2), (11.9, wide / 2), (11.9, -wide / 2), (7.9, -narrow / 2)],
                             facecolor=GREEN, edgecolor="black"))
        ax.text(9.9, 0, "Decoder\nsame matrix,\nflipped", ha="center", va="center", fontsize=14,
                fontweight="bold", color="white")
        ax.text(6.9, -1.55, "PCA: a linear autoencoder (no training, found with a formula)", ha="center",
                fontsize=14)
        ax.set_xlim(-0.2, 14.0)
        ax.set_ylim(-1.8, 1.3)
        ax.axis("off")
        fig.tight_layout()
        return fig

    def plot_pca_parts(mean, shapes, codes):
        """How the PCA decoder rebuilds a day: average day + code_i × shape_i."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.3), gridspec_kw={"width_ratios": [2, 3]})
        axes[0].plot(HOURS, mean, color="gray")
        axes[0].set_title("Average day")
        axes[0].set_ylabel("Temperature (°C)")
        colors = [BLUE, ORANGE, GREEN, PURPLE]
        for i, (shape, code) in enumerate(zip(shapes[:4], codes[:4])):
            axes[1].plot(HOURS, code * shape, color=colors[i], label=f"{code:+.1f} × shape {i + 1}")
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_title("+ the k numbers × fixed shapes" + (" (first 4 shown)" if len(codes) > 4 else ""))
        axes[1].set_ylabel("Change (°C)")
        axes[1].legend(fontsize=12, loc="best")
        for ax in axes:
            ax.set_xlabel("Hour of day")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_week(df, start):
        """Temperature and humidity for one week."""
        part = df.iloc[start:start + 7 * 24]
        fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
        axes[0].plot(part["time"], part["temperature"], color=VERMILLION)
        axes[0].set_ylabel("Temperature (°C)")
        axes[1].plot(part["time"], part["humidity"], color=BLUE)
        axes[1].set_ylabel("Humidity (%)")
        axes[0].set_title("A normal week of 'sensor' readings")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def plot_fault_example(clean, faulty, kind):
        """Temperature of one day without and with the injected fault."""
        fig, ax = plt.subplots(figsize=(11, 4.3))
        ax.plot(HOURS, clean[:, 0], "o-", color="gray", label="Normal reading")
        ax.plot(HOURS, faulty[:, 0], "o--", color=VERMILLION, label=f"With fault: {kind.lower()}")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_ae_diagram(latent, n_in):
        """Wide -> narrow -> wide: input, encoder, bottleneck, decoder, output."""
        fig, ax = plt.subplots(figsize=(13, 4))
        wide, narrow = 2.6, 0.35 + 1.6 * min(latent, 48) / 48
        boxes = [(0.0, wide, f"Input\n{n_in} numbers\n(24 h × 2)", "#DDDDDD"),
                 (6.1, narrow, f"Bottleneck\n{latent} number{'s' if latent > 1 else ''}", ORANGE),
                 (12.2, wide, f"Output\n{n_in} numbers\n(rebuilt day)", "#DDDDDD")]
        for x, h, text, color in boxes:
            ax.add_patch(FancyBboxPatch((x, -h / 2), 1.6, h, boxstyle="round,pad=0.05", facecolor=color,
                                        edgecolor="black"))
            ax.text(x + 0.8, 0, text, ha="center", va="center", fontsize=13)
        ax.add_patch(Polygon([(1.9, wide / 2), (5.9, narrow / 2), (5.9, -narrow / 2), (1.9, -wide / 2)],
                             facecolor=SKY, edgecolor="black"))
        ax.text(3.9, 0, "Encoder", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.add_patch(Polygon([(7.9, narrow / 2), (11.9, wide / 2), (11.9, -wide / 2), (7.9, -narrow / 2)],
                             facecolor=GREEN, edgecolor="black"))
        ax.text(9.9, 0, "Decoder", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        ax.text(6.9, -1.75, "Training goal: output = input", ha="center", fontsize=14)
        ax.set_xlim(-0.2, 14.0)
        ax.set_ylim(-2.0, 1.6)
        ax.axis("off")
        fig.tight_layout()
        return fig

    def plot_loss(history):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, history["train_loss"], "o-", color=BLUE, label="Training")
        ax.plot(epochs, history["val_loss"], "s--", color=GREEN, label="Validation")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction error")
        ax.set_title("Learning curve")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_reconstructions(pairs, titles):
        """Temperature of each (original, reconstruction) pair, side by side."""
        fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 4.3), sharey=True)
        for ax, (orig, rec), title in zip(np.atleast_1d(axes), pairs, titles):
            ax.plot(HOURS, orig[:, 0], "o-", color="gray", label="Input")
            ax.plot(HOURS, rec[:, 0], color=VERMILLION, label="Reconstruction")
            ax.set_title(title)
            ax.set_xlabel("Hour of day")
            ax.grid(True, alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("Temperature (°C)")
        np.atleast_1d(axes)[0].legend(fontsize=12)
        fig.tight_layout()
        return fig

    def plot_error_hist(train_err, test_err, labels, threshold):
        """Training errors (what normal looks like) and test days, with the threshold."""
        bins = np.logspace(np.log10(min(train_err.min(), test_err.min()) + 1e-6),
                           np.log10(max(train_err.max(), test_err.max())), 40)
        fig, ax = plt.subplots(figsize=(12, 4.6))
        for values, color, alpha, name in [(train_err, "#BBBBBB", 1.0, "Training windows (normal)"),
                                           (test_err[labels == 0], BLUE, 0.6, "Test days: normal"),
                                           (test_err[labels == 1], VERMILLION, 0.6, "Test days: with fault")]:
            ax.hist(values, bins=bins, color=color, alpha=alpha, label=name,
                    weights=np.full(len(values), 100 / max(1, len(values))))
        ax.axvline(threshold, color="black", linestyle="--", label="Threshold")
        ax.set_xscale("log")
        ax.set_xlabel("Reconstruction error (log scale)")
        ax.set_ylabel("Share of each group (%)")
        ax.legend(fontsize=12)
        fig.tight_layout()
        return fig

    def plot_flagged(times, days, labels, flags):
        """Test-period temperature, day by day: true faults marked, flagged days shaded."""
        fig, ax = plt.subplots(figsize=(14, 4.4))
        for i, t0 in enumerate(times):
            if flags[i]:
                ax.axvspan(i, i + 1, color=SKY, alpha=0.35, linewidth=0)
            ax.plot(i + HOURS / 24, days[i, :, 0], color=VERMILLION if labels[i] else "gray",
                    linewidth=1.6 if labels[i] else 1.2)
        ax.set_xlim(0, len(times))
        tick = np.arange(0, len(times), 14)
        ax.set_xticks(tick, [f"{times[i]:%b %d}" for i in tick])
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Test period: days with a fault in red, days flagged by the detector shaded blue")
        fig.tight_layout()
        return fig

    def plot_latent(z, color_values, labels):
        """Each test day as a point in the bottleneck (first two numbers or their 2-D projection)."""
        fig, ax = plt.subplots(figsize=(8, 6.2))
        sc = ax.scatter(z[labels == 0, 0], z[labels == 0, 1], c=color_values[labels == 0], cmap="viridis",
                        s=70, edgecolor="black", label="Normal day")
        ax.scatter(z[labels == 1, 0], z[labels == 1, 1], marker="X", s=130, color=VERMILLION, edgecolor="black",
                   label="Day with fault")
        fig.colorbar(sc, ax=ax, label="Mean humidity of the day (%)")
        ax.set_xlabel("Bottleneck number 1")
        ax.set_ylabel("Bottleneck number 2")
        ax.legend(fontsize=12, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    return (
        plot_ae_diagram,
        plot_error_hist,
        plot_fault_example,
        plot_flagged,
        plot_latent,
        plot_loss,
        plot_pca_day,
        plot_pca_diagram,
        plot_pca_parts,
        plot_reconstructions,
        plot_week,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        # Autoencoders: Finding Faulty Sensor Readings

        **Step 1:** a bottleneck by hand · **Step 2:** sensor data and faults · **Step 3:** train · **Step 4:**
        choose the threshold · **Step 5:** the bottleneck matters · **Step 6:** look inside
        """),
        mo.accordion({
            "More details: why learn without labels?": mo.md(r"""
            In the field we often have **years of normal readings** from a sensor, but almost **no labeled
            examples of faults**, which are rare and all different. An **autoencoder** skips labels entirely:
            it is trained to **reproduce its own input**, so the input itself is the training target
            (*self-supervised* learning). A network that has only seen normal days rebuilds normal days well,
            and rebuilds unusual days badly. That reconstruction error is what we use to flag problems.
            """)
        }),
    ])
    return


@app.cell
def _(ac, mo, rc):
    # Hourly Chachapoyas weather (the same data as the RNN notebook), used here as a "field sensor"
    try:
        sensor = ac.SensorDays(rc.load_weather(rc.fetch(rc.WEATHER_FILE)))
        sensor_error = None
    except Exception as _e:
        sensor, sensor_error = None, str(_e)
    mo.stop(
        sensor is None,
        mo.callout(mo.md(f"**Could not load the sensor data.** Check the internet connection.\n\n`{sensor_error}`"),
                   kind="danger"),
    )
    x_train = sensor.scale(sensor.sliding("train"))
    x_val = sensor.scale(sensor.sliding("validation", stride=24))
    test_days, test_starts = sensor.calendar_days("test")
    return sensor, test_days, test_starts, x_train, x_val


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 1: A bottleneck by hand

        Can we describe a whole day of temperature (24 numbers) with just a few numbers? A classic method,
        **PCA**, does it with one matrix: it is already a simple, *linear* autoencoder.
        """),
        mo.accordion({
            "More details: compression": mo.md(r"""
            Most days in Chachapoyas have a similar shape: cool at night, warm in the early afternoon. So we
            do not need all 24 hourly values to describe a day: a few numbers saying "how warm overall", "how
            big the daily swing" and so on are almost enough. Here we find those numbers with a classic method,
            **principal component analysis (PCA)**. Keeping **k** numbers and rebuilding the day from them shows
            the trade-off: with few numbers we keep the main shape and lose the details.
            """),
            "More details: PCA is a simple, linear autoencoder": mo.md(r"""
            PCA already has the three parts of an autoencoder (diagram below):

            - **Encoder:** one matrix multiplication turns the 24 hourly values into k numbers.
            - **Bottleneck:** the k numbers.
            - **Decoder:** the same matrix, flipped (transposed), turns the k numbers back into 24 values. The
              rebuilt day is the **average day plus k fixed shapes**, each multiplied by one of the k numbers
              (see "How the decoder rebuilds this day" below the plot).

            | | PCA | Neural autoencoder (Step 3) |
            |---|---|---|
            | Encoder and decoder | one matrix each (linear) | several layers with ReLU (non-linear) |
            | How it is found | a formula, no training | trained with gradient descent |
            | What it can rebuild | sums of fixed shapes | bent, more flexible curves |
            | Input here | temperature only (24 numbers) | temperature and humidity (48 numbers) |

            In fact, if we built the Step 3 network with **no ReLU** (only linear layers) and trained it to
            rebuild its input, at best it would keep the same information as PCA (the same k-number summary,
            possibly mixed differently among the k numbers). The ReLU layers are what let the autoencoder go
            beyond sums of fixed shapes.
            """),
        }),
    ])
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(1, 12, step=1, value=2, label="Numbers kept (k)")
    k_slider
    return (k_slider,)


@app.cell
def _(
    k_slider,
    mo,
    np,
    plot_pca_day,
    plot_pca_diagram,
    plot_pca_parts,
    sensor,
):
    # PCA of daily temperature curves (training days): keep k components, rebuild the day
    _days, _ = sensor.calendar_days("train")
    _temp = _days[:, :, 0]
    _mean = _temp.mean(axis=0)
    _, _, _vt = np.linalg.svd(_temp - _mean, full_matrices=False)
    _vt = _vt * np.sign(_vt[np.arange(len(_vt)), np.abs(_vt).argmax(axis=1)])[:, None]  # fixed signs

    def _encode(x, k):
        return (x - _mean) @ _vt[:k].T  # 24 numbers -> k numbers

    def _decode(z, k):
        return _mean + z @ _vt[:k]  # k numbers -> 24 numbers

    _k = k_slider.value
    _errors = [np.abs(_decode(_encode(_temp, k), k) - _temp).mean() for k in range(1, 13)]
    _day = _temp[100]
    _code = _encode(_day, _k)
    mo.vstack([
        plot_pca_diagram(_k),
        plot_pca_day(_day, _decode(_code, _k), _k, _errors),
        mo.accordion({"How the decoder rebuilds this day": plot_pca_parts(_mean, _vt[:_k], _code)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 2: The sensor data and the faults

        We treat the Chachapoyas weather as a field sensor and train only on **normal** days.
        """),
        mo.accordion({
            "More details: how the faults are created": mo.md(r"""
            The network learns from the training period (January 2023 to May 2024), which we assume is
            fault-free. To test it, we take the **109 days** of the test period (mid-September to December
            2024) and add a fault to the temperature of **half of them**, chosen at random. Because we created
            the faults ourselves, we know exactly which days are faulty, so we can score the detector honestly.
            The four fault types are typical of real field sensors:

            - **Stuck sensor:** the reading freezes at one value for several hours.
            - **Spike:** a single wrong reading, much too high.
            - **Drift:** the reading slowly creeps away from the true value during the day.
            - **Dropout to zero:** the sensor reports 0 for a few hours (for example, a loose cable).

            The **fault size** goes from 1 (subtle) to 5 (obvious).
            """)
        }),
    ])
    return


@app.cell
def _(plot_week, sensor):
    plot_week(sensor.df, 30 * 24)
    return


@app.cell
def _(ac, mo):
    fault_dropdown = mo.ui.dropdown(options=list(ac.FAULTS), value="Stuck sensor", label="Fault type")
    size_slider = mo.ui.slider(1, 5, step=1, value=2, label="Fault size")
    mo.hstack([fault_dropdown, size_slider], justify="start", gap=2)
    return fault_dropdown, size_slider


@app.cell
def _(ac, fault_dropdown, mo, np, plot_fault_example, size_slider, test_days):
    # Inject the chosen fault into half of the test days (fixed seed, so the demo is repeatable)
    faulty_days, fault_labels = ac.faulty_test_set(test_days, fault_dropdown.value, size_slider.value)
    example_day = int(np.flatnonzero(fault_labels)[0])
    mo.vstack([
        mo.md(f"*{fault_dropdown.value}: {ac.FAULTS[fault_dropdown.value]}.* "
              f"**{fault_labels.sum()}** of the {len(fault_labels)} test days have this fault."),
        plot_fault_example(test_days[example_day], faulty_days[example_day], fault_dropdown.value),
    ])
    return example_day, fault_labels, faulty_days


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 3: Train the autoencoder

        The **encoder** squeezes a day into a few numbers (the **bottleneck**); the **decoder** rebuilds the
        day from them. Same picture as PCA in Step 1, but now the encoder and decoder are non-linear layers
        learned by training.
        """),
        mo.accordion({
            "More details: the network": mo.md(r"""
            The input is one day: 24 hourly values of temperature and humidity, a 2 × 24 table (48 numbers).
            Training simply asks the output to match the input, measured by the average squared difference
            (the **reconstruction error**). No labels are used. The table below the diagram lists every layer:

            - **Encoder:** two 1D convolutional layers (filters sliding along the hours, as in the DNA example).
              The first finds 16 local patterns at every hour; the second, with **stride 2**, looks at every
              second hour, so the day shrinks from 24 to 12 steps while the patterns grow to 32. **Flatten**
              lays the 32 × 12 table out as one list of 384 numbers, and a **Linear** layer squeezes it into
              the bottleneck.
            - **Decoder:** the same steps in **reverse order**. A Linear layer grows the bottleneck back to 384
              numbers, **Unflatten** turns them into a 32 × 12 table, a **transposed convolution** with stride 2
              does the opposite of the stride-2 convolution (12 steps back to 24), and a last convolution
              returns the 2 channels: temperature and humidity.
            """),
            "More details: why a mirrored design?": mo.md(r"""
            The decoder has to undo what the encoder did, so it is natural to build it as the encoder **played
            backwards**: every layer that shrinks the data on the way in has a partner that grows it on the way
            out (Linear ↔ Linear, Flatten ↔ Unflatten, stride-2 convolution ↔ transposed convolution). Both
            halves end up with about the same number of parameters.

            Mirroring is a convenient habit, not a rule: the decoder could have a different number of layers.
            Note also that the mirror is only in the **shape**. In PCA (Step 1), the decoder used the very same
            matrix as the encoder, flipped. Here the decoder learns **its own weights**, which is part of what
            makes the autoencoder more flexible than PCA.
            """),
        }),
    ])
    return


@app.cell
def _(mo):
    latent_slider = mo.ui.slider(1, 48, step=1, value=4, label="Bottleneck size", debounce=True)
    epochs_slider = mo.ui.slider(5, 40, step=1, value=15, label="Epochs", debounce=True)
    mo.hstack([latent_slider, epochs_slider], justify="start", gap=2)
    return epochs_slider, latent_slider


@app.cell
def _(ac, epochs_slider, latent_slider, mo, plot_ae_diagram):
    _model = ac.SensorAE(latent_slider.value)

    def _shape(s):
        return " × ".join(str(v) for v in s)

    def _row(part, name, shape, n):
        if part == "Encoder" and shape == (latent_slider.value,):
            return f"| **Bottleneck** | **{name}** | **{_shape(shape)}** | **{n:,}** |"
        return f"| {part} | {name} | {_shape(shape)} | {n:,} |"

    _rows = "\n".join(_row(*r) for r in ac.layer_table(_model))
    _enc = ac.count_params(_model.encoder)
    _dec = ac.count_params(_model.decoder)
    mo.vstack([
        plot_ae_diagram(latent_slider.value, 48),
        mo.md(
            "| Part | Layer | Output shape (channels × hours) | Parameters |\n|---|---|---|---|\n"
            "| Input | one day | 2 × 24 | 0 |\n" + _rows
            + f"\n\n**Total parameters: {_enc + _dec:,}** (encoder {_enc:,}, decoder {_dec:,}). "
            "Read the shapes from top to bottom: they shrink to the bottleneck and grow back in mirror order."
        ),
        mo.md(f"*Estimated training time: about {max(2, round(0.3 * epochs_slider.value))} s*"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        **Train Model** trains a new autoencoder with the settings above; **Load Pre-trained** loads one
        trained longer beforehand.
        """),
        mo.accordion({
            "More details: live training vs. pre-trained": mo.md(r"""
            **Train Model** starts from random weights and uses the bottleneck size and epochs chosen above
            (a few seconds). **Load Pre-trained** loads an autoencoder with a bottleneck of **4** trained for
            40 epochs; the bottleneck slider is ignored in that case.
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
def _(ac):
    def load_pretrained(latent, fetch):
        """Load one pre-trained autoencoder (bottleneck 2, 4 or 48) and its training history."""
        import torch as _torch

        ckpt = _torch.load(fetch(ac.WEIGHTS_FILE), map_location="cpu", weights_only=False)
        entry = ckpt["models"][latent]
        model = ac.SensorAE(latent, len(ckpt["columns"]))
        model.load_state_dict(entry["state_dict"])
        return model, entry["history"]

    return (load_pretrained,)


@app.cell
def _(
    ac,
    epochs_slider,
    latent_slider,
    load_pretrained,
    mo,
    pretrained_button,
    rc,
    torch,
    train_button,
    x_train,
    x_val,
):
    mo.stop(
        not (train_button.value or pretrained_button.value),
        mo.md("*Press **Train Model** or **Load Pre-trained** to get a network.*"),
    )
    if pretrained_button.value:
        ae_model, ae_history = load_pretrained(4, rc.fetch)
        ae_source = f"pre-trained autoencoder (bottleneck 4, {len(ae_history['train_loss'])} epochs)"
    else:
        torch.manual_seed(0)
        ae_model = ac.SensorAE(latent_slider.value)
        with mo.status.spinner(title="Training the autoencoder..."):
            ae_history = ac.train_ae(ae_model, x_train, x_val, epochs=epochs_slider.value)
        ae_source = (f"autoencoder trained live (bottleneck {latent_slider.value}, {epochs_slider.value} epochs, "
                     f"{ae_history['seconds']:.0f} s)")
    train_errors = ac.reconstruction_error(ae_model, x_train)
    return ae_history, ae_model, ae_source, train_errors


@app.cell
def _(
    ac,
    ae_history,
    ae_model,
    ae_source,
    example_day,
    fault_labels,
    faulty_days,
    mo,
    np,
    plot_loss,
    plot_reconstructions,
    sensor,
):
    _normal = int(np.flatnonzero(fault_labels == 0)[0])
    _pairs = []
    for _day in (faulty_days[_normal], faulty_days[example_day]):
        _x = sensor.scale(_day[None])
        _pairs.append((_day, sensor.unscale(ac.reconstruct(ae_model, _x))[0]))
    mo.vstack([
        mo.md(f"**Results for the {ae_source}**"),
        plot_loss(ae_history),
        plot_reconstructions(_pairs, ["A normal day", "A day with a fault"]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **The key idea:** the network learned what a **normal** day looks like. On a faulty day it rebuilds
        the normal shape it expects, not the fault, so the difference between input and reconstruction is large.
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 4: Choosing the threshold

        Days the network cannot rebuild well are flagged. But how large an error is "too large"?
        """),
        mo.accordion({
            "More details: setting the threshold": mo.md(r"""
            The gray histogram shows the reconstruction errors on the **training** period, which we assume is
            normal. A common choice is to set the threshold so that only 1% of normal windows exceed it (the
            99th percentile). Moving the threshold is a trade-off:

            - **Lower threshold:** more faults found, but more **false alarms** on normal days.
            - **Higher threshold:** fewer false alarms, but more **missed faults**.

            The right choice depends on what each mistake costs in the field: a technician's trip for nothing,
            or a broken sensor feeding wrong data into irrigation or disease-risk decisions.
            """)
        }),
    ])
    return


@app.cell
def _(mo):
    percentile_slider = mo.ui.slider(90.0, 100.0, step=0.1, value=99.0,
                                     label="Threshold (percentile of training errors)")
    percentile_slider
    return (percentile_slider,)


@app.cell
def _(
    ac,
    ae_model,
    fault_labels,
    faulty_days,
    mo,
    np,
    percentile_slider,
    plot_error_hist,
    plot_flagged,
    sensor,
    test_starts,
    train_errors,
):
    _threshold = float(np.percentile(train_errors, percentile_slider.value))
    test_errors = ac.reconstruction_error(ae_model, sensor.scale(faulty_days))
    _flags = test_errors > _threshold
    _found = int((_flags & (fault_labels == 1)).sum())
    _missed = int((~_flags & (fault_labels == 1)).sum())
    _false = int((_flags & (fault_labels == 0)).sum())
    _quiet = int((~_flags & (fault_labels == 0)).sum())
    mo.vstack([
        plot_error_hist(train_errors, test_errors, fault_labels, _threshold),
        mo.hstack([
            mo.md(f"""
            | | Flagged | Not flagged |
            |---|---|---|
            | **Day with fault** | {_found} ✅ found | {_missed} ❌ missed |
            | **Normal day** | {_false} ⚠️ false alarm | {_quiet} ✅ |
            """),
            mo.md(f"**Faults found: {_found} of {_found + _missed}**<br>"
                  f"**False alarms: {_false} of {_false + _quiet} normal days**"),
        ], justify="start", gap=3, align="center"),
        plot_flagged(sensor.df["time"].iloc[test_starts].tolist(), faulty_days, fault_labels, _flags),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 5: The bottleneck matters

        The same faulty day, rebuilt by autoencoders with a bottleneck of **2**, **4** and **48** numbers.
        """),
        mo.accordion({
            "More details: too narrow, too wide": mo.md(r"""
            - **Too narrow (2):** the network cannot even rebuild normal days well, so normal errors are large,
              the threshold is high and **subtle faults are missed**.
            - **Too wide (48, as many numbers as the input):** the network has room to simply **copy** its input,
              faults included. Look at the rightmost panel: much of the fault is reproduced. In this clean data
              the wide network still flags many faults, because it rebuilds normal days almost perfectly and
              its threshold is extremely low; with a real, noisy sensor that small margin disappears and it would
              raise many false alarms.
            - **In between (4):** the network keeps the normal daily shape and **ignores the fault**, which is
              exactly what makes the fault visible.

            The bottleneck size is a knob, chosen with validation data, just like the number of epochs or filters.
            These three autoencoders were trained beforehand (40 epochs each).
            """)
        }),
    ])
    return


@app.cell
def _(
    ac,
    example_day,
    fault_labels,
    faulty_days,
    load_pretrained,
    mo,
    np,
    plot_reconstructions,
    rc,
    sensor,
    test_days,
    x_train,
):
    _pairs, _titles, _rows = [], [], []
    _x_test = sensor.scale(faulty_days)
    for _latent in (2, 4, 48):
        _model, _ = load_pretrained(_latent, rc.fetch)
        _rec = ac.reconstruct(_model, _x_test)
        _flags = ac.reconstruction_error(_model, _x_test) > np.percentile(
            ac.reconstruction_error(_model, x_train), 99)
        # Share of the fault the network reproduced (0 = ignored the fault, 1 = copied it exactly)
        _f = fault_labels == 1
        _clean, _faulty = sensor.scale(test_days[_f]), _x_test[_f]
        _copied = 1 - ((_rec[_f] - _faulty) ** 2).sum() / ((_clean - _faulty) ** 2).sum()
        _pairs.append((faulty_days[example_day], sensor.unscale(_rec[example_day])))
        _titles.append(f"Bottleneck {_latent}")
        _rows.append(f"| {_latent} | {int((_flags & _f).sum())} of {int(_f.sum())} | "
                     f"{int((_flags & ~_f).sum())} | {max(0.0, _copied):.0%} |")
    _n_fault, _n_normal = int((fault_labels == 1).sum()), int((fault_labels == 0).sum())
    mo.vstack([
        plot_reconstructions(_pairs, _titles),
        mo.md("| Bottleneck | Faults found | False alarms | Share of the fault copied |\n|---|---|---|---|\n"
              + "\n".join(_rows) + "\n\n*Threshold: 99th percentile of each network's training errors.*"),
        mo.accordion({
            "More details: what the columns mean": mo.md(f"""
            The table uses the fault type and size chosen in Step 2, on all {_n_fault + _n_normal} test days
            ({_n_fault} with a fault, {_n_normal} normal).

            - **Bottleneck:** how many numbers the encoder squeezes each day into.
            - **Faults found:** days with a fault whose reconstruction error is above the threshold (higher
              is better). The threshold is set separately for each network, at the 99th percentile of its own
              training errors, so the Step 4 slider does not change this table.
            - **False alarms:** normal days flagged anyway (lower is better), out of {_n_normal}.
            - **Share of the fault copied:** how much of the fault shows up in the rebuilt day. We compare the
              rebuilt day with the faulty input and with the true, fault-free day. **0%** means the rebuilt day
              matches the fault-free day: the network ignored the fault, which is what we want. **100%** means
              it matches the faulty input: the network copied the fault. This is the column that shows a wide
              bottleneck "just copying".
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 6: Look inside the bottleneck

        Each test day becomes a point: its bottleneck numbers.
        """),
        mo.accordion({
            "More details: the latent space": mo.md(r"""
            The bottleneck numbers are a compact summary of each day, learned without labels. Here we use the
            pre-trained autoencoder with a bottleneck of **2**, so each day is simply a point on a plane. Days
            with similar weather land close together: the color shows the day's mean humidity, and humid and
            dry days separate. Days with faults are marked with a red X. This kind of compact summary is also
            useful as input to other models (*feature extraction*), the same idea as the features learned by
            the CNN and RNN.

            *Looking ahead:* **transformers** (Part 5) also have an **encoder**: it turns each word, DNA letter
            or amino acid into a list of numbers (an **embedding**) that summarizes it in context, like the
            bottleneck numbers here. Many are trained much like the denoising autoencoder in the Colab notebook:
            some tokens of the input are hidden, and the model must fill them in. The protein model ESM-2 learns
            this way from millions of protein sequences, with no labels. One difference: a transformer encoder
            keeps one embedding **per token** instead of squeezing everything into one small bottleneck, so the
            hidden tokens, not a narrow bottleneck, are what stop it from simply copying.
            """)
        }),
    ])
    return


@app.cell
def _(
    fault_labels,
    faulty_days,
    load_pretrained,
    plot_latent,
    rc,
    sensor,
    torch,
):
    _model, _ = load_pretrained(2, rc.fetch)
    with torch.no_grad():
        _z = _model.encode(torch.as_tensor(sensor.scale(faulty_days))).numpy()
    plot_latent(_z, faulty_days[:, :, 1].mean(axis=1), fault_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to notice

    - An autoencoder is trained to **rebuild its own input** through a narrow **bottleneck**; no labels are needed.
    - Trained only on normal days, it rebuilds normal days well and faulty days badly: the **reconstruction
      error flags anomalies**, and the **threshold** trades missed faults against false alarms.
    - The **bottleneck size** matters: too narrow and normal days look unusual; too wide and the network can copy
      the faults.

    ## Questions for discussion

    1. How would you choose the threshold for a real sensor network in the field?
    2. What does a false alarm cost on a farm, compared with a missed fault?
    """)
    return


if __name__ == "__main__":
    app.run()
