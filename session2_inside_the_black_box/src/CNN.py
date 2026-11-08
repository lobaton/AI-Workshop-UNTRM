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
    import torch.nn.functional as F
    from sklearn.metrics import confusion_matrix

    # Shared model/data code lives in cnn_common.py next to this notebook; download it if missing (e.g. molab)
    try:
        import cnn_common as cc
    except ImportError:
        _cache = os.path.join(os.path.expanduser("~"), ".cache", "untrm_workshop")
        os.makedirs(_cache, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
            "session2_inside_the_black_box/src/cnn_common.py",
            os.path.join(_cache, "cnn_common.py"),
        )
        sys.path.insert(0, _cache)
        import cnn_common as cc

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return F, cc, confusion_matrix, mo, np, plt, torch


@app.cell(hide_code=True)
def _(cc, confusion_matrix, np, plt, torch):
    # HELPER FUNCTIONS (plotting)

    # Projection-friendly defaults: large fonts, thick lines
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "lines.linewidth": 3, "lines.markersize": 8,
    })
    TRAIN_COLOR, VAL_COLOR = "#0072B2", "#E69F00"  # Okabe-Ito, colorblind-safe

    KERNELS = {
        "Vertical edge": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32),
        "Horizontal edge": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32),
        "Blur": np.full((3, 3), 1 / 9, dtype=np.float32),
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    }

    def plot_filter_demo(gray, kernel, feature_map, kernel_name):
        """Input image | kernel numbers | feature map."""
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [4, 1.6, 4]})
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title(f"Input ({gray.shape[0]}×{gray.shape[1]})")
        axes[1].imshow(kernel, cmap="RdBu_r", vmin=-np.abs(kernel).max(), vmax=np.abs(kernel).max())
        for (r, c), v in np.ndenumerate(kernel):
            axes[1].text(c, r, f"{v:.2g}", ha="center", va="center", fontsize=14,
                         color="white" if abs(v) > 0.6 * np.abs(kernel).max() else "black")
        axes[1].set_title("Kernel (3×3)")
        # Raw convolution output (no activation): edge kernels give signed values, shown red (+) / blue (-)
        if "edge" in kernel_name:
            lim = np.abs(feature_map).max()
            axes[2].imshow(feature_map, cmap="RdBu_r", vmin=-lim, vmax=lim)
            axes[2].set_title(f"Feature map ({feature_map.shape[0]}×{feature_map.shape[1]})\n"
                              "red = positive, blue = negative", fontsize=14)
        else:
            axes[2].imshow(feature_map, cmap="gray")
            axes[2].set_title(f"Feature map ({feature_map.shape[0]}×{feature_map.shape[1]})")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig

    def plot_examples(images, labels, per_class=4):
        """A few training images per class."""
        fig, axes = plt.subplots(len(cc.CLASS_NAMES), per_class, figsize=(2.4 * per_class, 7.5))
        for c, name in enumerate(cc.CLASS_NAMES):
            idx = np.flatnonzero(labels == c)[:per_class]
            for j, i in enumerate(idx):
                axes[c, j].imshow(images[i])
                axes[c, j].set_xticks([])
                axes[c, j].set_yticks([])
            axes[c, 0].set_ylabel(name, fontsize=14)
        fig.tight_layout()
        return fig

    def plot_relu_demo(feature_map):
        """ReLU curve | signed feature map | feature map after ReLU."""
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [3, 4, 4]})
        z = np.linspace(-2, 2, 101)
        axes[0].axhline(0, color="gray", linewidth=1)
        axes[0].axvline(0, color="gray", linewidth=1)
        axes[0].plot(z, np.maximum(z, 0), color=TRAIN_COLOR)
        axes[0].set_xlabel("Input")
        axes[0].set_ylabel("Output")
        axes[0].set_title("ReLU")
        axes[0].grid(True, alpha=0.3)
        lim = np.abs(feature_map).max()
        axes[1].imshow(feature_map, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[1].set_title("Feature map\n(red = positive, blue = negative)", fontsize=14)
        axes[2].imshow(np.maximum(feature_map, 0), cmap="magma")
        axes[2].set_title("After ReLU\n(negatives set to 0)", fontsize=14)
        for ax in axes[1:]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig

    def plot_pool_demo(feature_map):
        """Feature map before and after 2×2 max pooling."""
        h, w = (feature_map.shape[0] // 2) * 2, (feature_map.shape[1] // 2) * 2
        pooled = feature_map[:h, :w].reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
        for ax, img, title in [(axes[0], feature_map, "Before pooling"), (axes[1], pooled, "After 2×2 max pooling")]:
            ax.imshow(img, cmap="magma")
            ax.set_title(f"{title}\n({img.shape[0]}×{img.shape[1]})", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig

    def plot_history(history):
        """Learning curves: loss and accuracy per epoch, training vs. validation."""
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs, history["train_loss"], "o-", color=TRAIN_COLOR, label="Training")
        axes[0].plot(epochs, history["val_loss"], "s--", color=VAL_COLOR, label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (cross-entropy)")
        axes[0].set_title("Learning curve: loss")
        axes[1].plot(epochs, 100 * np.array(history["train_acc"]), "o-", color=TRAIN_COLOR, label="Training")
        axes[1].plot(epochs, 100 * np.array(history["val_acc"]), "s--", color=VAL_COLOR, label="Validation")
        axes[1].axhline(100 / 3, color="gray", linestyle=":", label="Chance (33%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Learning curve: accuracy")
        axes[1].legend(loc="lower right", fontsize=12)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_confusion(y_true, y_pred):
        """Confusion matrix with counts, rows = true class."""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(cc.CLASS_NAMES)))
        names = [n.replace(" ", "\n", 1) for n in cc.CLASS_NAMES]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=18,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Test set: confusion matrix")
        fig.tight_layout()
        return fig

    def plot_first_layer_filters(model):
        """The learned 5×5 RGB filters of the first convolutional layer."""
        w = model.blocks[0][0].weight.detach().cpu().numpy()  # (F, 3, 5, 5)
        n = w.shape[0]
        cols = 8
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.7 * rows + 0.6))
        for k, ax in enumerate(np.atleast_1d(axes).ravel()):
            ax.set_xticks([])
            ax.set_yticks([])
            if k < n:
                f = w[k].transpose(1, 2, 0)
                ax.imshow((f - f.min()) / (f.max() - f.min() + 1e-8), interpolation="nearest")
            else:
                ax.axis("off")
        fig.suptitle("Layer 1 filters (5×5, color)", fontsize=16)
        fig.tight_layout()
        return fig

    def rank_channels(model, x, batch_size=128):
        """Per layer: channels sorted by mean activation over all images in x, and a color scale per channel.

        Using one fixed ranking and scale makes the feature maps comparable from one leaf to the next.
        """
        model.eval()
        sums, peaks = None, None
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                maps = model.feature_maps(x[i:i + batch_size])
                batch_sums = [m.mean(dim=(2, 3)).sum(0) for m in maps]
                batch_peaks = [m.amax(dim=(2, 3)) for m in maps]  # (batch, channels)
                sums = batch_sums if sums is None else [s + b for s, b in zip(sums, batch_sums)]
                peaks = batch_peaks if peaks is None else [torch.cat([p, b]) for p, b in zip(peaks, batch_peaks)]
        return [
            {"order": (s / len(x)).argsort(descending=True), "vmax": p.median(dim=0).values}
            for s, p in zip(sums, peaks)
        ]

    def plot_feature_maps(model, x, ranking, n_show=6):
        """Input image plus the feature maps of the n_show filters most active on the training set."""
        with torch.no_grad():
            maps = model.feature_maps(x[None])
        fig, axes = plt.subplots(2, n_show + 1, figsize=(2.3 * (n_show + 1), 5.2))
        for row, layer in enumerate([0, 1]):
            m = maps[layer][0]
            axes[row, 0].imshow(cc.to_image(x))
            axes[row, 0].set_title("Input", fontsize=13)
            for j, ch in enumerate(ranking[layer]["order"][:n_show]):
                axes[row, j + 1].imshow(m[ch].numpy(), cmap="magma", vmin=0,
                                        vmax=float(ranking[layer]["vmax"][ch]) + 1e-6)
                axes[row, j + 1].set_title(f"Filter {int(ch) + 1}", fontsize=13)
            axes[row, 0].set_ylabel(f"Layer {layer + 1}\n({m.shape[1]}×{m.shape[2]})", fontsize=14)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        return fig

    def receptive_fields(model):
        """Receptive field, in input pixels, of one value in each block's feature map (after ReLU, before pooling)."""
        rf, jump, sizes = 1, 1, []
        for block in model.blocks:
            conv, pool = block[0], block[2]
            rf += (conv.kernel_size[0] - 1) * jump
            sizes.append(rf)
            rf += (pool.kernel_size - 1) * jump
            jump *= pool.stride
        return sizes

    def plot_receptive_field_diagram(n=7, gap=3.2):
        """Stacked 3×3-conv layers: one cell on top sees 3×3 below, which sees 5×5 in the input."""
        from matplotlib.patches import Polygon

        skew_x, skew_y = 0.55, 0.38  # perspective: rows recede up and to the right

        def corner(i, j, y0):
            return (j + skew_x * i, y0 + skew_y * i)

        layers = [  # (name, half-width of the highlighted square around the center cell, color)
            ("Input image: 5×5", 2, "#E69F00"),
            ("Layer 1: 3×3", 1, "#56B4E9"),
            ("Layer 2: 1 neuron", 0, "#D55E00"),
        ]
        c = n // 2
        fig, ax = plt.subplots(figsize=(9, 6.2))
        for k, (name, h, color) in enumerate(layers):
            y0 = k * gap
            for i in range(n):
                for j in range(n):
                    inside = abs(i - c) <= h and abs(j - c) <= h
                    pts = [corner(i, j, y0), corner(i, j + 1, y0), corner(i + 1, j + 1, y0), corner(i + 1, j, y0)]
                    ax.add_patch(Polygon(pts, closed=True, facecolor=color if inside else "#F2F2F2",
                                         edgecolor="#555555", linewidth=1))
            ax.text(n + skew_x * n + 0.4, y0 + skew_y * n / 2, name, va="center", fontsize=15)
        # Dashed lines from the corners of each highlighted patch down to the patch it depends on
        for k in range(1, len(layers)):
            h_top, h_bot = layers[k][1], layers[k - 1][1]
            for di in (0, 1):
                for dj in (0, 1):
                    top = corner(c - h_top + di * (2 * h_top + 1), c - h_top + dj * (2 * h_top + 1), k * gap)
                    bot = corner(c - h_bot + di * (2 * h_bot + 1), c - h_bot + dj * (2 * h_bot + 1), (k - 1) * gap)
                    ax.plot([top[0], bot[0]], [top[1], bot[1]], color="#333333", linewidth=1.2, linestyle="--")
        ax.set_xlim(-0.3, n + skew_x * n + 5.5)
        ax.set_ylim(-0.3, (len(layers) - 1) * gap + skew_y * n + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Without pooling: two stacked 3×3 convolutions")
        fig.tight_layout()
        return fig

    def plot_receptive_fields_on_leaf(image, sizes):
        """Squares showing how many input pixels one neuron of each conv layer of our network depends on."""
        import matplotlib.patheffects as pe

        # Deliberately different colors from the no-pooling diagram (these are different layers); bright
        # colors with a black outline so they stand out against the green leaf
        colors = ["#FFFFFF", "#FFE119", "#FF3EF2"]
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.imshow(image)
        cy, cx = image.shape[0] / 2, image.shape[1] / 2
        for k, (s, col) in enumerate(zip(sizes, colors)):
            rect = plt.Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, edgecolor=col, linewidth=3.5,
                                 label=f"Conv layer {k + 1}: {s}×{s} pixels")
            rect.set_path_effects([pe.Stroke(linewidth=7, foreground="black"), pe.Normal()])
            ax.add_patch(rect)
        ax.legend(loc="upper left", fontsize=13, framealpha=0.9, facecolor="#555555", labelcolor="white")
        ax.set_title("With pooling: our network\n(conv → pool → conv → pool → conv)", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    return (
        KERNELS,
        plot_confusion,
        plot_examples,
        plot_feature_maps,
        plot_filter_demo,
        plot_first_layer_filters,
        plot_history,
        plot_pool_demo,
        plot_receptive_field_diagram,
        plot_receptive_fields_on_leaf,
        plot_relu_demo,
        rank_channels,
        receptive_fields,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        # Convolutional Networks: Learning to See Leaf Disease

        **Step 1:** one filter by hand · **Step 2:** train a small CNN on bean leaves · **Step 3:** look
        inside the network
        """),
        mo.accordion({
            "More details: where this fits": mo.md(r"""
            In Part 1 we saw that a neural network **learns its own features**: its hidden layers re-draw the
            data so that the classes become easy to separate. Images need a special kind of layer to do this
            well: the **convolutional layer**. In this notebook we will:

            1. Apply **one filter by hand** to a bean-leaf photo, to see what a convolution does.
            2. **Train a small CNN** to recognize two bean diseases and healthy leaves.
            3. **Look inside** the trained network to see the filters it learned on its own.
            """)
        }),
    ])
    return


@app.cell
def _(cc, mo):
    # Load the bean-leaf images (128×128 pixels), from the local data/ folder or GitHub
    try:
        beans = cc.load_beans(cc.fetch(cc.BEANS_FILE))
        load_error = None
    except Exception as _e:
        beans, load_error = None, str(_e)
    mo.stop(
        beans is None,
        mo.callout(mo.md(f"**Could not load the bean-leaf images.** Check the internet connection.\n\n`{load_error}`"),
                   kind="danger"),
    )
    x_train_np, y_train_np = beans["train"]
    x_val_np, y_val_np = beans["validation"]
    x_test_np, y_test_np = beans["test"]
    return x_test_np, x_train_np, x_val_np, y_test_np, y_train_np, y_val_np


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 1: One filter by hand

        A small filter (kernel) slides over the image and lights up where its pattern appears.
        """),
        mo.accordion({
            "More details: how a convolution works": mo.md(r"""
            A **convolution** slides a small grid of numbers, the **kernel** (or filter), across the image. At
            each position it multiplies the kernel with the pixels underneath and adds them up. A large result
            means "the pattern this kernel looks for is here". The output image is called a **feature map**.

            Two ideas make this efficient:

            - **Local receptive field**: each output value only looks at a small 3×3 patch, not the whole image.
            - **Weight sharing**: the *same* 9 numbers are reused at every position of the image.

            Pick a kernel and a leaf below. The **stride** is how many pixels the kernel jumps each step: a
            larger stride gives a smaller feature map.
            """)
        }),
    ])
    return


@app.cell
def _(KERNELS, cc, mo):
    kernel_dropdown = mo.ui.dropdown(options=list(KERNELS), value="Vertical edge", label="Kernel")
    leaf_dropdown = mo.ui.dropdown(options=cc.CLASS_NAMES, value="bean rust", label="Leaf")
    stride_slider = mo.ui.slider(1, 4, step=1, value=1, label="Stride")
    mo.hstack([kernel_dropdown, leaf_dropdown, stride_slider], justify="start", gap=2)
    return kernel_dropdown, leaf_dropdown, stride_slider


@app.cell
def _(
    F,
    KERNELS,
    cc,
    kernel_dropdown,
    leaf_dropdown,
    np,
    plot_filter_demo,
    stride_slider,
    torch,
    x_train_np,
    y_train_np,
):
    # Grayscale version of the first training image of the chosen class
    _img = x_train_np[np.flatnonzero(y_train_np == cc.CLASS_NAMES.index(leaf_dropdown.value))[0]]
    _gray = _img.mean(axis=2) / 255.0
    _kernel = KERNELS[kernel_dropdown.value]

    # The convolution itself: slide the 3×3 kernel over the image
    _fmap = F.conv2d(
        torch.from_numpy(_gray).float()[None, None],
        torch.from_numpy(_kernel)[None, None],
        stride=stride_slider.value,
    )[0, 0].numpy()

    plot_filter_demo(_gray, _kernel, _fmap, kernel_dropdown.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Try this": mo.callout(
            mo.md(r"""
            With the *vertical edge* kernel, strong red or blue lines appear where the brightness changes from
            left to right, such as leaf veins and the leaf border: red where it goes from bright to dark, blue
            where it goes from dark to bright. Switch to *horizontal edge* and different lines light up. In a CNN we do not choose
            these numbers: **the network learns its own kernels** from the training examples.
            """),
            kind="info",
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 2: Train a small CNN on bean leaves

        Field photos of bean leaves in three classes: **angular leaf spot**, **bean rust** and **healthy**.
        """),
        mo.accordion({
            "More details: the beans dataset": mo.md(r"""
            The **beans** dataset (Makerere AI Lab, Uganda) has photos of bean leaves taken in the field with
            a phone. It is the same dataset as Session 1, here reduced to 128×128 pixels so it trains quickly.
            The data comes already split into a **training** set (to learn from), a **validation** set (to
            watch progress during training) and a **test** set (the final exam).
            """)
        }),
    ])
    return


@app.cell
def _(mo, plot_examples, x_test_np, x_train_np, x_val_np, y_train_np):
    mo.vstack([
        mo.md(
            f"**Training:** {len(x_train_np)} images &nbsp;&nbsp; **Validation:** {len(x_val_np)} "
            f"&nbsp;&nbsp; **Test:** {len(x_test_np)}"
        ),
        plot_examples(x_train_np, y_train_np),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### Two more building blocks: ReLU and pooling

        After each convolution, **ReLU** keeps only the positive responses, and **max pooling** shrinks the
        feature map by keeping the strongest response in each small block.
        """),
        mo.accordion({
            "What does ReLU do?": mo.md(r"""
            **ReLU** (Rectified Linear Unit) is a very simple rule applied to every number in the feature map:
            negative values become **0**, positive values stay as they are.

            - It turns each feature map into a clean answer to "is my pattern here, and how strongly?"
              Negative responses (for example, an edge going the opposite way) are simply ignored.
            - It is the **activation function** that makes the network non-linear. Without it, stacking many
              layers would be no more powerful than a single layer. The MLP in Part 1 used the same ReLU.
            """),
            "What does max pooling do?": mo.md(r"""
            **Max pooling** splits the feature map into small blocks (here 2×2) and keeps only the largest
            value of each block:

            ```
            1 3 | 0 2
            5 2 | 1 1          5  2
            ----+----    →
            0 4 | 8 6          4  8
            1 2 | 3 7
            ```

            - Width and height are halved, so there are **4× fewer numbers** to process.
            - If a spot moves by a pixel, the result barely changes: what matters is **whether** the pattern
              is present, not its exact position.
            - After pooling, the next layer's 5×5 filter covers a **larger region** of the leaf.
            """),
        }, multiple=True),
    ])
    return


@app.cell
def _(
    F,
    KERNELS,
    cc,
    kernel_dropdown,
    leaf_dropdown,
    mo,
    np,
    plot_pool_demo,
    plot_relu_demo,
    torch,
    x_train_np,
    y_train_np,
):
    # Same leaf and kernel as in Step 1 (stride 1), then ReLU and 2×2 max pooling
    _img = x_train_np[np.flatnonzero(y_train_np == cc.CLASS_NAMES.index(leaf_dropdown.value))[0]]
    _fmap = F.conv2d(
        torch.from_numpy(_img.mean(axis=2) / 255.0).float()[None, None],
        torch.from_numpy(KERNELS[kernel_dropdown.value])[None, None],
    )[0, 0].numpy()

    mo.vstack([
        mo.md(f"*Using the **{kernel_dropdown.value}** kernel on the **{leaf_dropdown.value}** leaf from Step 1.*"),
        plot_relu_demo(_fmap),
        plot_pool_demo(np.maximum(_fmap, 0)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### The architecture

        Our CNN stacks three blocks of **convolution → ReLU → max pooling**, followed by a small **fully
        connected** layer that makes the decision.
        """),
        mo.accordion({
            "Why not just use the MLP from Part 1?": mo.md(r"""
            Flattening a 128×128 color image gives 49,152 numbers. A single fully connected layer with 1,000
            neurons would need about **49 million weights**, and it would treat neighboring pixels as
            unrelated. The CNN re-uses each small filter everywhere, so it needs only a tiny fraction of that
            (compare the totals in the table below).
            """)
        }),
    ])
    return


@app.cell
def _(mo):
    filters_dropdown = mo.ui.dropdown(options=["8", "16"], value="16", label="Filters in the first layer")
    n_train_slider = mo.ui.slider(150, 1034, step=1, value=600, label="Training images", debounce=True)
    epochs_slider = mo.ui.slider(3, 15, step=1, value=8, label="Epochs", debounce=True)
    mo.hstack([filters_dropdown, n_train_slider, epochs_slider], justify="start", gap=2)
    return epochs_slider, filters_dropdown, n_train_slider


@app.cell
def _(epochs_slider, filters_dropdown, mo, n_train_slider):
    # Rough CPU cost per image per epoch, measured on 2 CPU threads
    _sec_per_image = {"8": 0.002, "16": 0.0067}[filters_dropdown.value]
    _estimate = _sec_per_image * n_train_slider.value * epochs_slider.value
    mo.md(f"*Estimated training time: about {_estimate:.0f} s*"
          + (" (over a minute: consider fewer images or epochs for the live demo)" if _estimate > 60 else ""))
    return


@app.cell
def _(cc, filters_dropdown, mo):
    _model = cc.BeansCNN(int(filters_dropdown.value))
    _rows = "\n".join(f"| {name} | {shape} | {n:,} |" for name, shape, n in cc.layer_table(_model))
    _mlp_weights = 128 * 128 * 3 * 1000 + 1000
    mo.md(
        "| Layer | Output shape (channels, height, width) | Parameters |\n|---|---|---|\n" + _rows
        + f"\n\n**Total CNN parameters: {cc.count_params(_model):,}** &nbsp; vs. &nbsp; one fully connected"
        f" layer of 1,000 neurons on the flattened image: **{_mlp_weights:,}**"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### Training

        **Train Model** trains the network live; **Load Pre-trained** loads the same network trained longer
        beforehand. The pre-trained network is shown when the notebook opens.
        """),
        mo.accordion({
            "More details: live training vs. pre-trained": mo.md(r"""
            **Train Model** trains a new network from random weights on the chosen number of training images
            (about 30 seconds with the default settings). Because we only train for a few epochs live, the
            result is modest. **Load Pre-trained** loads the *same* network that was trained beforehand for 40
            epochs on all training images, with flipped and rotated copies of the leaves to make it more
            robust.
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
    cc,
    epochs_slider,
    filters_dropdown,
    mo,
    n_train_slider,
    pretrained_button,
    torch,
    train_button,
    x_train_np,
    x_val_np,
    y_train_np,
    y_val_np,
):

    # Pre-trained by default (when the notebook opens or Load Pre-trained is pressed); Train Model trains live
    pretrained_button  # pressing it re-runs this cell
    if not train_button.value:
        _ckpt = torch.load(cc.fetch(cc.WEIGHTS_FILE), map_location="cpu", weights_only=False)
        cnn_model = cc.BeansCNN(_ckpt["n_filters"])
        cnn_model.load_state_dict(_ckpt["state_dict"])
        cnn_history = _ckpt["history"]
        model_source = f"pre-trained network ({len(cnn_history['train_loss'])} epochs, all training images)"
    else:
        torch.manual_seed(0)
        cnn_model = cc.BeansCNN(int(filters_dropdown.value))
        _xs, _ys = cc.random_subset(x_train_np, y_train_np, n_train_slider.value)
        with mo.status.spinner(title="Training the CNN..."):
            cnn_history = cc.train_model(
                cnn_model, cc.to_tensor(_xs), torch.from_numpy(_ys),
                cc.to_tensor(x_val_np), torch.from_numpy(y_val_np),
                epochs=epochs_slider.value,
            )
        model_source = (f"network trained live ({epochs_slider.value} epochs, {len(_ys)} images, "
                        f"{cnn_history['seconds']:.0f} s)")
    return cnn_history, cnn_model, model_source


@app.cell
def _(
    cc,
    cnn_history,
    cnn_model,
    mo,
    model_source,
    plot_confusion,
    plot_history,
    torch,
    x_test_np,
    y_test_np,
):
    _, test_acc, _test_preds = cc.evaluate(cnn_model, cc.to_tensor(x_test_np), torch.from_numpy(y_test_np))
    mo.vstack([
        mo.md(f"**Results for the {model_source}**"),
        plot_history(cnn_history),
        mo.hstack([
            plot_confusion(y_test_np, _test_preds),
            mo.vstack([
                mo.md(f"""
                ### Test accuracy: {test_acc:.0%}
                Chance level (guessing): 33%
                """),
                mo.accordion({
                    "How to read the matrix": mo.md(r"""
                    Each row is the true class, each column the network's answer. Numbers on the diagonal
                    are correct; the others show which diseases get confused.
                    """)
                }),
            ]),
        ], align="center", widths=[3, 2]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Learn more: reading the learning curves": mo.md(r"""
        - The **loss** measures how wrong the network is; it should go down as training proceeds.
        - The **validation** curve uses images the network does not learn from. If training accuracy keeps
          rising while validation accuracy stalls or drops, the network is **overfitting**: memorizing the
          training images instead of learning the disease patterns (as in Part 1).
        - The **test** accuracy is measured only once, at the end, on images never used for any decision.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 3: Look inside the network

        The **filters the network learned** in its first layer, starting from random numbers.
        """),
        mo.accordion({
            "More details: learned filters": mo.md(r"""
            Nobody programmed these filters: they started as random numbers and were adjusted during
            training. Each tile is one 5×5 color filter. Compare them with the hand-made kernels from Step 1:
            many become edge detectors or color detectors (for example, brown-yellow spots vs. green leaf).
            """)
        }),
    ])
    return


@app.cell
def _(cnn_model, plot_first_layer_filters):
    plot_first_layer_filters(cnn_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        One test leaf passed through the network: bright = where a filter responds.
        """),
        mo.accordion({
            "More details: feature maps": mo.md(r"""
            Each image below is a **feature map**: the output of one filter (after ReLU) for the chosen leaf.
            Bright areas are where that filter's pattern was found. We show the 6 filters of layer 1 and
            layer 2 that are **most active on average over all training images**. The same filters, in the
            same order and with the same brightness scale, are shown for every leaf, so you can compare how
            each filter reacts to healthy and diseased leaves. Layer 2 maps are half the size because of max
            pooling. The network's answer for this leaf is shown underneath.
            """)
        }),
    ])
    return


@app.cell
def _(cc, cnn_model, mo, rank_channels, x_train_np):
    # Rank each layer's filters once per model by their mean activation over the training images
    with mo.status.spinner(title="Ranking filters on the training images..."):
        channel_ranking = rank_channels(cnn_model, cc.to_tensor(x_train_np))
    return (channel_ranking,)


@app.cell
def _(cc, mo, np, y_test_np):
    _options = {}
    for _c, _name in enumerate(cc.CLASS_NAMES):
        for _k, _i in enumerate(np.flatnonzero(y_test_np == _c)[:4]):
            _options[f"{_name} #{_k + 1}"] = int(_i)
    test_image_dropdown = mo.ui.dropdown(options=_options, value="bean rust #1", label="Test leaf")
    test_image_dropdown
    return (test_image_dropdown,)


@app.cell
def _(
    cc,
    channel_ranking,
    cnn_model,
    mo,
    plot_feature_maps,
    test_image_dropdown,
    torch,
    x_test_np,
):
    _x = cc.to_tensor(x_test_np[test_image_dropdown.value][None])[0]
    cnn_model.eval()
    with torch.no_grad():
        _probs = torch.softmax(cnn_model(_x[None]), dim=1)[0].numpy()
    _pred = " · ".join(f"{name}: {p:.0%}" for name, p in zip(cc.CLASS_NAMES, _probs))
    mo.vstack([plot_feature_maps(cnn_model, _x, channel_ranking), mo.md(f"**Network's answer:** {_pred}")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            **Edges → textures → parts → objects:** each layer builds on the patterns found by the one before.
            """),
            kind="info",
        ),
        mo.accordion({
            "More details: the feature hierarchy": mo.md(r"""
            Layer 1 responds to simple things such as edges and colors. Layer 2 combines them into
            **textures**, such as the rust-colored spots, and because of pooling it looks at a larger region of
            the leaf. Much deeper networks continue this hierarchy up to whole objects. This is the same idea
            as in Part 1: **the network learns its own features**, now organized in space.
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### How much of the leaf does one neuron see?

        Each neuron looks at a small patch of the layer below, but stacking layers makes that patch grow:
        deeper layers see a **larger part of the leaf**.
        """),
        mo.accordion({
            "More details: the receptive field": mo.md(r"""
            The region of the input image that one neuron depends on is its **receptive field** (or
            *support*.

            **Left, without pooling:** every layer uses a 3×3 kernel. One neuron in layer 2 looks at a
            3×3 patch of layer 1, and each of those 9 neurons looks at its own 3×3 patch of the image.
            Together they cover a **5×5** patch of the input. Every extra 3×3 layer adds only 2 more pixels.

            **Right, with pooling (our network):** after a 2×2 max pooling, each step of the next filter
            moves 2 pixels in the original image, so the field grows much faster. Our network uses a 5×5
            filter, pooling, a 5×5 filter, pooling, then a 3×3 filter: one neuron of its three conv layers
            sees 5×5, 14×14 and 24×24 pixels. That is enough for a rust spot but not the whole leaf. Much
            deeper networks stack enough layers to see the entire image.
            """)
        }),
    ])
    return


@app.cell
def _(
    cc,
    mo,
    plot_receptive_field_diagram,
    plot_receptive_fields_on_leaf,
    receptive_fields,
    test_image_dropdown,
    x_test_np,
):
    mo.hstack([
        plot_receptive_field_diagram(),
        plot_receptive_fields_on_leaf(x_test_np[test_image_dropdown.value], receptive_fields(cc.BeansCNN())),
    ], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to notice

    - A convolution is a **small filter sliding over the image**; it lights up where its pattern appears.
    - A CNN **learns its own filters** from labeled examples, and stacked layers build **edges → textures →
      shapes**.
    - CNNs use **far fewer weights** than a fully connected network on images, but they still need many
      labeled examples to work well.

    ## Questions for discussion

    1. These photos were taken in the field in Uganda. Would the model work on bean or coffee leaves
       photographed in Amazonas? What could go wrong with a model trained only on lab photos?
    2. What would a useful filter look for in a microscopy image of spores or nematodes?

    *Want to explore more?* Try the interactive [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
    from Georgia Tech.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "References": mo.md(r"""
        - **Convolutional networks:** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based
          learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
          [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
        - **Deep CNNs with ReLU, max pooling and data augmentation (AlexNet):** Krizhevsky, A., Sutskever, I., &
          Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in
          Neural Information Processing Systems 25* (NIPS 2012). Republished in *Communications of the ACM*,
          60(6), 84–90 (2017). [doi:10.1145/3065386](https://doi.org/10.1145/3065386)
        - **ReLU:** Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann
          machines. *Proceedings of the 27th International Conference on Machine Learning (ICML 2010)*,
          807–814.
        - **Global average pooling:** Lin, M., Chen, Q., & Yan, S. (2014). Network in network. *ICLR 2014*.
          [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
        - **Adam optimizer:** Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
          *ICLR 2015*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
        - **Bean-leaf images:** Makerere AI Lab, in collaboration with the National Crops Resources Research
          Institute (NaCRRI), Uganda (2020). *ibean: bean disease dataset* (healthy, angular leaf spot, bean
          rust; MIT license). [github.com/AI-Lab-Makerere/ibean](https://github.com/AI-Lab-Makerere/ibean);
          also on Hugging Face as `AI-Lab-Makerere/beans`.
        - **PyTorch** (the library used to build and train the networks): Paszke, A., et al. (2019). PyTorch:
          An imperative style, high-performance deep learning library. *Advances in Neural Information
          Processing Systems 32* (NeurIPS 2019). [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
        """)
    })
    return


if __name__ == "__main__":
    app.run()
