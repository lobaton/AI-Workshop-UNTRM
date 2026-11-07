import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import io
    import os
    import sys
    import urllib.request
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # Shared code lives next to this notebook; download it if missing (e.g. molab)
    try:
        import attention_common as at
    except ImportError:
        _cache = os.path.join(os.path.expanduser("~"), ".cache", "untrm_workshop")
        os.makedirs(_cache, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
            "session2_inside_the_black_box/src/attention_common.py",
            os.path.join(_cache, "attention_common.py"),
        )
        sys.path.insert(0, _cache)
        import attention_common as at

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return at, io, mo, np, plt, torch


@app.cell(hide_code=True)
def _(at, np, plt):
    # HELPER FUNCTIONS (plotting)
    from matplotlib.patches import Rectangle

    # Projection-friendly defaults: large fonts, thick lines
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "lines.linewidth": 3, "lines.markersize": 9,
    })
    # Okabe-Ito colorblind-safe palette
    BLUE, ORANGE, SKY, VERMILLION, GREEN, PURPLE = "#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"
    LETTER_COLORS = {"A": GREEN, "C": BLUE, "G": ORANGE, "T": VERMILLION}

    def plot_toy_attention(words, matrix, row, output, features, show_scores):
        """Score or weight matrix | the chosen word's row | the chosen word's output."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [5, 3, 2.2]})
        ax = axes[0]
        if show_scores:
            ax.imshow(matrix, cmap="Greys")
            ax.set_title("Scores: Q·Kᵀ / √d")
        else:
            ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1)
            ax.set_title("Attention weights (each row sums to 100%)")
        ax.add_patch(Rectangle((-0.5, row - 0.5), len(words), 1, fill=False, edgecolor=VERMILLION, linewidth=3))
        ax.set_xticks(range(len(words)), words, rotation=60, ha="right")
        ax.set_yticks(range(len(words)), words)
        ax.set_xlabel("Word being looked at (key)")
        ax.set_ylabel("Word that looks (query)")

        ax = axes[1]
        values = matrix[row]
        ax.barh(range(len(words)), values, color=[VERMILLION if v == values.max() else "#BBBBBB" for v in values])
        ax.set_yticks(range(len(words)), words)
        ax.invert_yaxis()
        ax.set_title(f"Row of '{words[row]}'")
        ax.set_xlabel("Score" if show_scores else "Weight")
        if not show_scores:
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

        ax = axes[2]
        ax.barh(range(len(features)), output, color=BLUE)
        ax.set_yticks(range(len(features)), features)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(f"Output of '{words[row]}'")
        ax.set_xlabel("Value")
        for a in axes[1:]:
            a.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_order_summary(summaries, labels, use_position):
        """Sentence summaries (average of the attention outputs) for two word orders."""
        fig, ax = plt.subplots(figsize=(10, 4))
        width = 0.38
        x = np.arange(len(summaries[0]))
        for i, (s, label, color) in enumerate(zip(summaries, labels, [BLUE, ORANGE])):
            ax.bar(x + (i - 0.5) * width, s, width, color=color, label=label)
        ax.set_xticks(x, [f"number {i + 1}" for i in x])
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Value")
        same = np.allclose(summaries[0], summaries[1])
        ax.set_title(("With" if use_position else "Without") + " positional encoding: the two summaries are "
                     + ("IDENTICAL" if same else "different"))
        ax.legend(fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_positional_encoding(n_positions=64, d=16):
        pe = at.sinusoidal_encoding(n_positions, d).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 2]})
        im = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_xlabel("Position in the sequence")
        axes[0].set_ylabel("Encoding number")
        axes[0].set_title("Each position gets its own pattern (column)")
        fig.colorbar(im, ax=axes[0])
        for i, color in zip([0, 4, 10], [VERMILLION, BLUE, GREEN]):
            axes[1].plot(pe[:, i], color=color, label=f"number {i + 1}")
        axes[1].set_xlabel("Position in the sequence")
        axes[1].set_title("Waves of different speeds")
        axes[1].legend(fontsize=12, loc="lower left")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def draw_sequence(ax, seq, starts, motif_len, y=0.0, label=""):
        """Draw a DNA sequence as colored letters, with boxes around motif A (orange) and motif B (blue)."""
        text = at.to_text(seq)
        for i, c in enumerate(text):
            ax.text(i + 0.5, y, c, ha="center", va="center", fontsize=13, family="monospace",
                    color=LETTER_COLORS[c], fontweight="bold")
        for p, color in zip(starts, [ORANGE, BLUE]):
            ax.add_patch(Rectangle((p, y - 0.4), motif_len, 0.8, facecolor=color, alpha=0.25, edgecolor=color,
                                   linewidth=2))
        ax.text(-0.5, y, label, ha="right", va="center", fontsize=14)

    def plot_dna_examples(seqs, starts, labels):
        fig, ax = plt.subplots(figsize=(16, 1.0 + 0.9 * len(seqs)))
        for i, (s, p, l) in enumerate(zip(seqs, starts, labels)):
            draw_sequence(ax, s, p, len(at.MOTIF_A), y=-i,
                          label="A before B (label 1)" if l else "B before A (label 0)")
        ax.set_xlim(-0.5, seqs.shape[1] + 0.5)
        ax.set_ylim(-len(seqs) + 0.4, 0.6)
        ax.axis("off")
        ax.set_title(f"Orange: motif A ({at.MOTIF_A}, the -35 box) · Blue: motif B ({at.MOTIF_B}, the -10 box)",
                     fontsize=14)
        fig.tight_layout()
        return fig

    def plot_learning(histories):
        fig, ax = plt.subplots(figsize=(9, 4))
        for (name, h), color in zip(histories.items(), [BLUE, ORANGE, "gray"]):
            ax.plot(np.arange(1, len(h["val_acc"]) + 1), h["val_acc"], "o-", color=color, label=name)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.5, label="Chance")
        ax.set_ylim(0.4, 1.02)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_dna_attention(weights, seq, starts, head):
        """Attention map of one head (tokens x tokens) | the rows of the two motif tokens."""
        n = weights.shape[0]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.35]})
        ax = axes[0]
        ax.imshow(weights, cmap="Greys", vmin=0, vmax=min(1.0, weights.max()))
        for p, color in zip(starts, [ORANGE, BLUE]):
            ax.axvline(p, color=color, linewidth=2)
            ax.axhline(p, color=color, linewidth=2)
        ax.set_xlabel("Token being looked at")
        ax.set_ylabel("Token that looks")
        ax.set_title(f"Head {head}: attention map")
        ax = axes[1]
        for p, color, name in zip(starts, [ORANGE, BLUE], ["motif A token", "motif B token"]):
            ax.axvspan(p - 0.5, p + 0.5, color=color, alpha=0.3)
            ax.plot(np.arange(n), weights[p], color=color, label=f"Where the {name} looks")
        ax.set_xlabel("Token position (each token = 6 letters starting here)")
        ax.set_ylabel("Attention weight")
        ax.set_title("Rows of the two motif tokens")
        ax.legend(fontsize=12, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_patches(img, patch=16):
        n = img.shape[0] // patch
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.5]})
        axes[0].imshow(img)
        for k in range(1, n):
            axes[0].axhline(k * patch - 0.5, color="white", linewidth=1.5)
            axes[0].axvline(k * patch - 0.5, color="white", linewidth=1.5)
        axes[0].set_title(f"{img.shape[0]}×{img.shape[1]} leaf → {n * n} patches of {patch}×{patch}")
        axes[0].axis("off")
        axes[1].axis("off")
        axes[1].set_title("The first 16 patches, as a sequence of tokens")
        for i in range(16):
            r, c = divmod(i, n)
            sub = axes[1].inset_axes([(i % 8) / 8 + 0.01, 0.55 - 0.45 * (i // 8), 1 / 8 - 0.02, 0.36])
            sub.imshow(img[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch])
            sub.set_xticks([])
            sub.set_yticks([])
            sub.set_xlabel(str(i + 1), fontsize=12)
        return fig

    def plot_esm_attention(attention, true, title):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
        axes[0].imshow(attention, cmap="Greys", vmin=0, vmax=np.percentile(attention, 99))
        axes[0].set_title(title)
        axes[1].imshow(true, cmap="Greys")
        axes[1].set_title("True contacts (crystal structure)")
        for ax in axes:
            ax.set_xlabel("Amino acid position")
            ax.set_ylabel("Amino acid position")
        fig.tight_layout()
        return fig

    def plot_contacts(pred, true, title, min_sep=6):
        """Upper triangle: the model's top-L predicted contacts; lower triangle: true contacts."""
        n = len(true)
        iu = np.triu_indices(n, min_sep)
        top = np.argsort(-pred[iu])[:n]
        pi, pj = iu[0][top], iu[1][top]
        hit = true[pi, pj] == 1
        ti, tj = np.nonzero(np.triu(true, min_sep))
        fig, ax = plt.subplots(figsize=(11.5, 7))
        ax.scatter(ti, tj, s=30, color="#BBBBBB", label="True contact (below the line)")
        ax.scatter(pj[hit], pi[hit], s=45, color=BLUE, label="Predicted, correct (above)")
        ax.scatter(pj[~hit], pi[~hit], s=45, color=VERMILLION, marker="x", label="Predicted, wrong (above)")
        ax.plot([0, n], [0, n], color="black", linewidth=1)
        ax.set_xlim(-1, n)
        ax.set_ylim(n, -1)
        ax.set_xlabel("Amino acid position")
        ax.set_ylabel("Amino acid position")
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        return fig

    return (
        plot_contacts,
        plot_dna_attention,
        plot_dna_examples,
        plot_esm_attention,
        plot_learning,
        plot_order_summary,
        plot_patches,
        plot_positional_encoding,
        plot_toy_attention,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        # Transformers: Attention for Words, DNA and Proteins

        **Step 1:** attention by hand · **Step 2:** word order · **Step 3:** a tiny transformer on DNA ·
        **Step 4:** tokens beyond words · **Step 5:** scale
        """),
        mo.accordion({
            "More details: why a new architecture?": mo.md(r"""
            The recurrent network of Part 3 reads a sequence **one step at a time**, carrying a memory forward.
            Information from far back has to survive many steps, and the steps cannot run in parallel. A
            **transformer** looks at the **whole sequence at once**: every element (a **token**) can directly
            look at every other token, however far apart, and decide how relevant it is. This mechanism is
            called **attention**. It is the core of large language models, and also of protein models such
            as ESM and AlphaFold.
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 1: Attention by hand

        Every word asks every other word: *how relevant are you to me?*
        """),
        mo.accordion({
            "More details: query, key and value": mo.md(r"""
            Each word (token) gets three short lists of numbers:

            - a **query** (Q): *what am I looking for?*
            - a **key** (K): *what am I?*
            - a **value** (V): *what do I pass on if someone looks at me?*

            The **score** between two words is their query · key (multiply matching numbers and add up),
            divided by √d, where d is the length of the lists (here 4) to keep the scores in a good range.
            A **softmax** turns each row of scores into **weights** that are positive and add up to 100%.
            The **output** of a word is the weighted mix of all the values.

            In a real transformer, the numbers come from the word's embedding multiplied by learned matrices
            (Q = X·W_Q, K = X·W_K, V = X·W_V). Here we **write them by hand** with four readable features, so
            you can check the result. In a real model, "it" also learns about "tired" or "wide" in an earlier
            layer; here we give "it" that hint directly in its query.
            """),
            "More details: tensor shapes": mo.md(r"""
            With 11 words and lists of 4 numbers: Q, K and V each have shape **(11, 4)**. The scores
            Q·Kᵀ have shape **(11, 11)**: one row per word that looks, one column per word looked at. The
            weights have the same shape, and the output, weights · V, is again **(11, 4)**: one new list of
            numbers per word, now mixed with its context.
            """),
        }),
    ])
    return


@app.cell
def _(np):
    # Hand-made numbers, 4 readable features per word
    FEATURES = ["living thing", "place", "pronoun", "describing word"]

    def toy_sentence(last_word):
        words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", last_word]
        K = np.zeros((11, 4))  # key: "what I am"
        K[1], K[5], K[7], K[10] = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        Q = np.zeros((11, 4))  # query: "what I am looking for"
        Q[1], Q[5] = K[1], K[5]
        Q[7] = [1, 0, 0, 0] if last_word == "tired" else [0, 1, 0, 0]  # 'it' looks for what can be tired / wide
        Q[10] = [0, 0, 1, 0]  # 'tired' / 'wide' looks for the word it describes
        V = K.copy()  # value: what each word passes on
        return words, 6 * Q, K, V

    def softmax_rows(S):
        e = np.exp(S - S.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    return FEATURES, softmax_rows, toy_sentence


@app.cell
def _(mo):
    last_word = mo.ui.dropdown(["tired", "wide"], value="tired", label="Last word")
    head_radio = mo.ui.radio(["Head 1: what does it refer to?", "Head 2: previous word"],
                             value="Head 1: what does it refer to?", label="Attention head")
    temperature = mo.ui.slider(0.2, 3.0, step=0.1, value=1.0, label="Temperature")
    look_word = mo.ui.dropdown({"animal": 1, "street": 5, "it": 7, "tired / wide": 10}, value="it",
                               label="Show the row of")
    show_radio = mo.ui.radio(["Weights (after softmax)", "Scores (before softmax)"],
                             value="Weights (after softmax)", label="Show")
    mo.hstack([mo.vstack([last_word, look_word, temperature]), head_radio, show_radio], justify="start", gap=3)
    return head_radio, last_word, look_word, show_radio, temperature


@app.cell
def _(
    FEATURES,
    head_radio,
    last_word,
    look_word,
    mo,
    np,
    plot_toy_attention,
    show_radio,
    softmax_rows,
    temperature,
    toy_sentence,
):
    _words, _Q, _K, _V = toy_sentence(last_word.value)
    if head_radio.value.startswith("Head 1"):
        _scores = _Q @ _K.T / np.sqrt(4)  # (11, 11)
    else:
        _scores = np.zeros((11, 11))
        _scores[np.arange(1, 11), np.arange(10)] = 3.0  # each word looks one position back (uses position, Step 2)
    _weights = softmax_rows(_scores / temperature.value)  # (11, 11), rows add up to 1
    _output = _weights @ _V  # (11, 4)
    _row = look_word.value
    _show_scores = show_radio.value.startswith("Scores")
    _top = int(_weights[_row].argmax())
    mo.vstack([
        plot_toy_attention(_words, _scores if _show_scores else _weights, _row, _output[_row], FEATURES,
                           _show_scores),
        mo.md(f"**'{_words[_row]}'** puts **{_weights[_row, _top]:.0%}** of its attention on "
              f"**'{_words[_top]}'**. Shapes: Q, K, V (11, 4) → scores and weights (11, 11) → output (11, 4)."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **Try:** switch the last word from *tired* to *wide* and watch where "it" looks. Lower the temperature
        to sharpen the attention; raise it to spread it out. **Two heads** look for different things at the
        same time (*multi-head attention*).
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 2: Word order

        Attention alone does not know **where** each word is: "dog bites man" and "man bites dog" look the same.
        """),
        mo.accordion({
            "More details: why attention is order-blind": mo.md(r"""
            The scores depend only on the words' numbers, not on their positions. If we shuffle the words,
            every word still finds the same partners with the same weights; only the rows and columns of the
            attention map are shuffled. So the **summary** of the sentence (the average of all outputs, which
            is what our DNA classifier in Step 3 uses) is exactly the same for "dog bites man" and "man bites
            dog".

            The fix is **positional encoding**: before attention, we add to each word's numbers a pattern
            that depends only on its position (below). Now the same word at a different position has
            different numbers, and the order matters.

            *Remember the RNN notebook:* we gave the hour of the day as sine and cosine, like a clock hand, so
            that 23:00 and 00:00 are close. Positional encoding uses the same trick with **several clock hands
            turning at different speeds**, so that every position gets its own pattern and nearby positions
            have similar patterns.
            """),
        }),
    ])
    return


@app.cell
def _(mo):
    position_switch = mo.ui.switch(value=False, label="Add positional encoding")
    position_switch
    return (position_switch,)


@app.cell
def _(at, np, plot_order_summary, position_switch, softmax_rows):
    # Three words with simple embeddings, and fixed random query/key matrices
    _emb = {"dog": [1, 0, 0, 0], "man": [0, 1, 0, 0], "bites": [0, 0, 1, 0]}
    _rng = np.random.default_rng(0)
    _Wq, _Wk = _rng.normal(size=(4, 4)), _rng.normal(size=(4, 4))
    _pe = at.sinusoidal_encoding(3, 4).numpy()

    def _summary(words):
        x = np.array([_emb[w] for w in words], dtype=float)
        if position_switch.value:
            x = x + _pe  # add the position pattern to each word
        weights = softmax_rows((x @ _Wq) @ (x @ _Wk).T / 2)
        return (weights @ x).mean(axis=0)  # average of the outputs = sentence summary

    plot_order_summary([_summary(["dog", "bites", "man"]), _summary(["man", "bites", "dog"])],
                       ["dog bites man", "man bites dog"], position_switch.value)
    return


@app.cell
def _(mo, plot_positional_encoding):
    mo.accordion({"Show the positional encoding": plot_positional_encoding()})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 3: A tiny transformer on DNA

        Does motif A come **before** motif B? A question about order and distance, which a CNN cannot answer.
        """),
        mo.accordion({
            "More details: the task": mo.md(rf"""
            In bacterial **promoters** (the switch in front of a gene), the **-35 box** (`TTGACA`, motif A)
            comes before the **-10 box** (`TATAAT`, motif B). Every sequence here contains both boxes once,
            10 to 24 letters apart, at random positions. In half of the sequences they are in promoter order
            (A before B, label 1); in the other half the order is reversed (label 0). Which motifs are present
            gives no clue: only their **order** does.

            In the DNA motif Colab notebook (Part 2), a 1D CNN found a motif with a filter and **global max
            pooling**, which keeps the best score of each filter and forgets **where** it was found. Here
            that is exactly the problem: the CNN can see both motifs but not which comes first, so it should
            stay near chance (50%). The transformer compares every position with every other position, and
            with positional encoding it knows where each token is.

            **Tokens:** the transformer reads the DNA as **overlapping 6-letter words** (6-mers), as the
            DNABERT model does: positions 1-6, 2-7, 3-8, and so on. Each motif then appears as one token.
            """),
        }),
    ])
    return


@app.cell
def _(at, mo, plot_dna_examples):
    # 3,000 training, 500 validation and 1,000 test sequences (fixed seed, so the demo is repeatable)
    dna = at.make_dataset()
    _seqs, _labels, _starts = dna["test"]
    _pick = [int((_labels == 1).argmax()), int((_labels == 0).argmax())]
    mo.vstack([
        mo.md(f"Every sequence has {_seqs.shape[1]} letters. Two examples from the test set:"),
        plot_dna_examples(_seqs[_pick], _starts[_pick], _labels[_pick]),
    ])
    return (dna,)


@app.cell
def _(mo):
    use_position = mo.ui.switch(value=True, label="Positional encoding")
    dna_epochs = mo.ui.slider(5, 40, step=1, value=15, label="Epochs", debounce=True)
    mo.hstack([use_position, dna_epochs], justify="start", gap=3)
    return dna_epochs, use_position


@app.cell
def _(at, dna_epochs, mo, use_position):
    _model = at.TinyTransformer(use_position=use_position.value)
    _rows = "\n".join(f"| {name} | {' × '.join(map(str, shape))} | {n:,} |"
                      for name, shape, n in at.layer_table(_model))
    mo.vstack([
        mo.md("| Layer | Output shape (tokens × numbers) | Parameters |\n|---|---|---|\n" + _rows
              + f"\n\n**Total: {at.count_params(_model):,} parameters**, almost all of them in the embedding table "
              "(32 numbers for each of the 4,096 possible 6-mers)."),
        mo.md(f"*Estimated training time: about {max(3, round(0.45 * dna_epochs.value))} s*"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        **Train Model** trains the transformer and the CNN live; **Load Pre-trained** loads both trained
        longer beforehand.
        """),
        mo.accordion({
            "More details: the network and the training": mo.md(r"""
            Each 6-mer token is turned into 32 numbers (the **embedding**), and the positional encoding is
            added. One **self-attention** layer with 2 heads lets every token look at every other token; a
            small **feed-forward** network then processes each token on its own. "Add & normalize" means the
            layer's input is added back to its output (a shortcut that makes training easier) and rescaled.
            Finally the tokens are **averaged** into one summary of the sequence, and a linear layer decides
            between "A before B" and "B before A". This is one **transformer encoder layer**; large models
            stack dozens of them.

            The CNN is the one from the DNA motif notebook: 4 filters 8 letters long, global max pooling and
            a linear layer. **Load Pre-trained** loads both models trained for 40 epochs, with or without
            positional encoding, following the switch above.
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
def _(at):
    def load_pretrained(name, fetch):
        """Load one pre-trained model ('transformer', 'transformer_no_position' or 'cnn') and its history."""
        import torch as _torch

        entry = _torch.load(fetch(at.WEIGHTS_FILE), map_location="cpu", weights_only=False)["models"][name]
        if name == "cnn":
            model = at.MotifCNN()
        else:
            model = at.TinyTransformer(use_position=(name == "transformer"))
        model.load_state_dict(entry["state_dict"])
        return model, entry["history"]

    return (load_pretrained,)


@app.cell
def _(
    at,
    dna,
    dna_epochs,
    load_pretrained,
    mo,
    pretrained_button,
    torch,
    train_button,
    use_position,
):
    mo.stop(
        not (train_button.value or pretrained_button.value),
        mo.md("*Press **Train Model** or **Load Pre-trained** to get the networks.*"),
    )
    _x, _y, _ = dna["train"]
    _xv, _yv, _ = dna["validation"]
    if pretrained_button.value:
        tt_model, tt_history = load_pretrained("transformer" if use_position.value else "transformer_no_position",
                                               at.fetch)
        cnn_model, cnn_history = load_pretrained("cnn", at.fetch)
        dna_source = "pre-trained models (40 epochs)"
    else:
        with mo.status.spinner(title="Training the transformer and the CNN..."):
            torch.manual_seed(0)
            tt_model = at.TinyTransformer(use_position=use_position.value)
            tt_history = at.train_model(tt_model, _x, _y, _xv, _yv, epochs=dna_epochs.value, lr=3e-3)
            torch.manual_seed(0)
            cnn_model = at.MotifCNN()
            cnn_history = at.train_model(cnn_model, _x, _y, _xv, _yv, epochs=dna_epochs.value, lr=1e-2)
        dna_source = (f"models trained live ({dna_epochs.value} epochs, "
                      f"{tt_history['seconds'] + cnn_history['seconds']:.0f} s)")
    return cnn_history, cnn_model, dna_source, tt_history, tt_model


@app.cell
def _(
    at,
    cnn_history,
    cnn_model,
    dna,
    dna_source,
    mo,
    plot_learning,
    tt_history,
    tt_model,
    use_position,
):
    _xt, _yt, _ = dna["test"]
    _tt_name = "Transformer" + ("" if use_position.value else " (no positional encoding)")
    mo.vstack([
        mo.md(
            f"**Test accuracy, {dna_source}** (1,000 sequences never seen in training)\n\n"
            "| Model | Test accuracy |\n|---|---|\n| Chance (guessing) | 50% |\n"
            f"| 1D CNN (Part 2) | {at.accuracy(cnn_model, _xt, _yt):.1%} |\n"
            f"| {_tt_name} | **{at.accuracy(tt_model, _xt, _yt):.1%}** |"
        ),
        plot_learning({_tt_name: tt_history, "1D CNN": cnn_history}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### Look inside: the attention map

        Which tokens does the transformer look at? The orange and blue lines mark the two motif tokens.
        """),
        mo.accordion({
            "More details: reading the attention map": mo.md(r"""
            Each row is one token that "looks", each column a token being looked at; darker means more
            attention. A dark **column** means many tokens look at that position. The plot on the right shows
            only the two motif rows. The network was never told where the motifs are, yet it learns to focus
            on them. In the models we trained, the two heads **split the work**: one looks almost only at
            motif A, the other at motif B (which head takes which motif changes from one training run to the
            next). Each token's output then carries the numbers of both motifs, including their positional
            encodings, so the last layers can compare the two positions. Heads in large models are usually
            much harder to interpret than this.
            """)
        }),
    ])
    return


@app.cell
def _(mo):
    dna_head = mo.ui.radio(["Head 1", "Head 2"], value="Head 1", label="Head")
    dna_example = mo.ui.slider(1, 20, step=1, value=1, label="Test sequence")
    mo.hstack([dna_head, dna_example], justify="start", gap=3)
    return dna_example, dna_head


@app.cell
def _(
    at,
    dna,
    dna_example,
    dna_head,
    mo,
    plot_dna_attention,
    tt_model,
):
    _xt, _yt, _st = dna["test"]
    _p = at.predict(tt_model, _xt)
    _i = dna_example.value - 1
    _head = int(dna_head.value[-1])
    _weights = at.attention_maps(tt_model, _xt[_i])[_head - 1]
    _truth = "A before B" if _yt[_i] else "B before A"
    _guess = "A before B" if _p[_i] > 0.5 else "B before A"
    _on_a, _on_b = _weights[:, _st[_i, 0]].mean(), _weights[:, _st[_i, 1]].mean()
    mo.vstack([
        mo.md(f"Test sequence {_i + 1}: truly **{_truth}**; the transformer says **{_guess}** "
              f"({max(_p[_i], 1 - _p[_i]):.0%} confident). On average, head {_head} puts **{_on_a:.0%}** of "
              f"each token's attention on motif A and **{_on_b:.0%}** on motif B."),
        plot_dna_attention(_weights, _xt[_i], _st[_i], _head),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 4: Tokens beyond words

        A token can be a word, a DNA 6-mer, a patch of an image, or an amino acid in a protein.
        """),
        mo.accordion({
            "More details: images as tokens": mo.md(r"""
            A **Vision Transformer** cuts an image into small square patches and treats them as a sequence of
            tokens, each with a positional encoding that says where the patch was. Attention then lets every
            patch look at every other patch, so a spot on one side of the leaf can be related to a spot on the
            other side from the very first layer. A CNN (Part 2) only sees small neighborhoods at first and
            needs many layers to cover the whole leaf.
            """)
        }),
    ])
    return


@app.cell
def _(at, io, np, plot_patches):
    from PIL import Image

    # One bean-rust leaf from the CNN notebook's data
    _pack = np.load(at.fetch("data/beans128.npz"))
    _i = int(np.flatnonzero(_pack["labels"] == 1)[0])
    _jpeg = _pack["jpeg"][_pack["offsets"][_i]:_pack["offsets"][_i + 1]].tobytes()
    plot_patches(np.asarray(Image.open(io.BytesIO(_jpeg)).convert("RGB")))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### Proteins: what does a protein language model look at?

        **ESM-2** read millions of protein sequences, one amino acid per token. Without ever seeing a
        structure, some of its attention heads follow which amino acids **touch** in the folded protein.
        Here: **ubiquitin** (76 amino acids).
        """),
        mo.accordion({
            "More details: how ESM-2 learned": mo.md(r"""
            *Remember the autoencoder's encoder?* ESM-2 is a transformer **encoder** trained much like the
            denoising autoencoder: some amino acids in each sequence are **hidden**, and the model must fill
            them in from the rest. No labels and no structures were used, only sequences. To fill in a hidden
            amino acid well, it helps to know which other amino acids **touch it in the folded protein**, so
            some attention heads end up following the 3D contacts. Unlike our autoencoder, ESM-2 keeps one
            embedding per amino acid instead of one small bottleneck.

            This is the idea of **pre-training**: train once on huge data, then reuse or adapt the model for
            many tasks (structure, function, effects of mutations). DNABERT does the same for DNA and Vision
            Transformers for images. Session 3 shows how to use such pre-trained models in code.

            The maps were computed beforehand for three sizes of ESM-2 and are loaded from the workshop
            repository. **True contacts:** pairs of amino acids closer than 8 Å in the crystal structure
            (PDB 1UBQ). The attention map shown is averaged in both directions (i looks at j, and j looks at
            i). AlphaFold builds on the same idea, with much more data and a module that turns the contacts
            into 3D coordinates. Try the hands-on Colab notebook to run ESM-2 on your own protein and explore
            every layer and head.
            """)
        }),
    ])
    return


@app.cell
def _(at, np):
    esm = dict(np.load(at.fetch(at.ESM_FILE)))
    return (esm,)


@app.cell
def _(esm, mo):
    esm_model = mo.ui.dropdown(
        {f"ESM-2, {p / 1e6:.0f}M parameters": i for i, p in enumerate(esm["params"])},
        value=f"ESM-2, {esm['params'][0] / 1e6:.0f}M parameters", label="Model size",
    )
    esm_model
    return (esm_model,)


@app.cell
def _(esm, esm_model, mo, plot_contacts, plot_esm_attention):
    _i = esm_model.value
    _layer, _head = (int(v) + 1 for v in esm["best_head"][_i])
    _att = esm["best_head_attention"][_i].astype(float)
    _n = len(esm["true_contacts"])
    mo.vstack([
        plot_esm_attention((_att + _att.T) / 2, esm["true_contacts"], f"Attention: layer {_layer}, head {_head}"),
        mo.md(f"The head of this model whose map looks most like the true contacts (most other heads look at "
              f"other things). Sequence: `{esm['sequence']}`"),
        mo.md("**Predicting contacts.** ESM-2 can also combine all its heads into a contact prediction:"),
        plot_contacts(esm["contacts"][_i].astype(float), esm["true_contacts"],
                      f"{_n} most likely contacts: {esm['precision'][_i]:.0%} correct"),
        mo.md("Above the diagonal: the model's 76 most likely contacts (at least 6 positions apart along the "
              "chain); below: the true contacts. Larger models find many more of them."),
    ])
    return


@app.cell(hide_code=True)
def _(esm, mo):
    _p = esm["params"]
    mo.vstack([
        mo.md(rf"""
        ## Step 5: Scale

        The same building block, stacked and scaled up.

        | Model | Parameters | What it does |
        |---|---|---|
        | Our DNA transformer (Step 3) | 140 thousand | one layer, A before B |
        | ESM-2 small (Step 4) | {_p[0] / 1e6:.1f} million | proteins, 6 layers |
        | ESM-2 medium | {_p[1] / 1e6:.0f} million | proteins, 12 layers |
        | ESM-2 large | {_p[2] / 1e6:.0f} million | proteins, 30 layers |
        | GPT-2 small (2019) | 124 million | text, 12 layers |
        | ESM-2 largest | 15 billion | proteins, 48 layers |
        | GPT-3 (2020) | 175 billion | text, 96 layers |
        | Today's largest language models | hundreds of billions or more | text, images, code (many sizes are not published) |
        """),
        mo.accordion({
            "More details: what changes with scale": mo.md(r"""
            The core operation does not change: attention, feed-forward, add & normalize, repeated layer after
            layer. What grows is the number of layers, the length of the embeddings, the number of heads, and
            above all the amount of training data and computing power. Step 4 showed one effect: the same
            protein model, made larger, predicts contacts much better. To see a full language model (GPT-2)
            running in your browser, try the **Transformer Explainer**:
            [poloclub.github.io/transformer-explainer](https://poloclub.github.io/transformer-explainer/).
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to notice

    - **Attention** lets every token score every other token (query · key), turn the scores into weights
      (softmax), and take a weighted mix of their values, all in parallel.
    - Attention alone is **order-blind**; **positional encoding** tells the model where each token is. With it,
      a one-layer transformer relates two motifs far apart, which a CNN with pooling cannot.
    - **Tokens** can be words, DNA k-mers, image patches or amino acids. Models **pre-trained** on huge data,
      such as ESM-2, learn useful structure (like protein contacts) without being told.

    ## Questions for discussion

    1. What would a **token** be in your own data (field images, sensor readings, sequences)?
    2. Why can a transformer relate two distant motifs more easily than the RNN from Part 3?
    """)
    return


if __name__ == "__main__":
    app.run()
