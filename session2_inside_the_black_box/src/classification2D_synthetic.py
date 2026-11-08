import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt

    return make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification in 2D: How Neural Networks Learn Features

    Imagine you are given a map covered in colored dots, and your job is to teach a computer to guess the
    color of any new dot just from its position. This is called **classification**.

    The catch: in this example, the colors are arranged so that **no straight line can split them apart**.
    Simple methods that draw straight dividing lines will fail.

    In this notebook you will see how a **neural network** solves the problem. Instead of drawing curvy lines
    directly, it learns to *re-draw the map*: it moves the dots to new positions where the colors **can** be
    split with straight lines. These new positions are called **features**, and learning good features is
    the key idea behind modern AI.

    We will go step by step:

    1. Create a dataset that is hard to classify.
    2. Build a small neural network.
    3. Train it and check how well it does.
    4. Peek *inside* the network to see the features it learned.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: The data

    Each dot below is one **data point**. It is described by just two numbers, its horizontal and vertical
    position (*Feature 1* and *Feature 2*). Its color is the **class** (the answer we want the computer to
    predict):

    - **Class 0** and **Class 1**: two interlocking half-moons in the middle.
    - **Class 2**: a ring that surrounds both moons.

    Use the slider to change how many points are generated for each class.
    """)
    return


@app.cell
def _(mo):
    # Slider to control number of samples per class
    samples_slider = mo.ui.slider(50, 500, step=10, value=200, label="Samples per class")
    samples_slider
    return (samples_slider,)


@app.cell
def data_generation(make_moons, np, plt, samples_slider):
    def _generate_dataset(n):
        # Two interleaving moons (already two classes); make_moons splits n_samples across both
        X_moons, y_moons = make_moons(n_samples=2 * n, noise=0.1, random_state=42)

        # Annulus (third class) that encloses the moons
        r_inner, r_outer = 2.0, 3.0
        theta = np.random.rand(n) * 2 * np.pi
        r = np.sqrt(np.random.rand(n) * (r_outer**2 - r_inner**2) + r_inner**2)
        X_annulus = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        y_annulus = np.full(n, 2)

        # Combine
        X = np.vstack((X_moons, X_annulus))
        y = np.concatenate((y_moons, y_annulus))
        return X, y

    # Generate data based on slider value
    X, y = _generate_dataset(samples_slider.value)

    # Plot
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title("Three‑class dataset: moons + annulus")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Class")

    # Return UI element and the plot
    plt.gca()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            **Try this:** imagine drawing straight lines on the plot to separate the three colors.
            You can't do it: the moons hook into each other, and the ring wraps around everything.
            We say this data is **not linearly separable**.
            """),
            kind="info",
        ),
        mo.accordion({
            "Learn more: why straight lines are not enough": mo.md(r"""
            Many classic methods (for example, *logistic regression*) can only draw straight boundaries
            between classes. They work well when the groups sit in different regions of the map, but here
            any straight line would cut through at least one moon or through the ring.

            We could hand-design new measurements to help, such as "distance from the center", which would
            pick out the ring. But that requires a human to understand the data first. A neural network
            discovers useful measurements like this **on its own**.
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Holding back an "exam"

    Before training, we set aside **20%** of the points as a **test set**. The network never sees these
    points while learning. Afterwards we use them as an exam to check that it truly learned the pattern,
    rather than just memorizing the points it was shown (the **training set**).
    """)
    return


@app.cell
def _(X, mo, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mo.md(f"**Training points:** {len(X_train)} &nbsp;&nbsp; **Test points:** {len(X_test)}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Step 2: The neural network

        A neural network is built from **layers** of simple units called **neurons**. Each neuron combines
        the numbers it receives into a single new number, a bit like a tiny detector that responds to a
        particular pattern. Stacking layers lets the network build complex patterns out of simple ones.

        Our network passes each point through these layers:

        | Layer | Size | What it does |
        |---|---|---|
        | Input | 2 | The point's position (Feature 1, Feature 2) |
        | Hidden layer 1 | 16 neurons | Detects simple patterns in the position |
        | Hidden layer 2 | 16 neurons | Combines those into more complex patterns |
        | **Hidden layer 3** | **2 neurons** | **The learned features**, squeezed down to 2 numbers |
        | Output | 3 | One score per class; the highest score is the prediction |

        After each hidden layer there is a **ReLU** step, which keeps positive values and turns negative
        values into zero. This simple "on/off" behavior is what lets the network bend and fold the map
        instead of only stretching it.

        Notice the deliberate **bottleneck**: the last hidden layer has only **2** neurons. That means
        every point is turned into just two new numbers, which we can **plot** later to see exactly how
        the network has re-drawn the map.
        """),
        mo.accordion({
            "Learn more: reading the model summary below": mo.md(r"""
            The printout below lists the layers in order:

            - `Linear(in_features=2, out_features=16)` is a layer of 16 neurons, each taking a weighted
              sum of the 2 inputs. The **weights** are the knobs the network adjusts while learning.
            - `ReLU()` is the "keep positives, zero out negatives" step.
            - The final `Linear(in_features=2, out_features=3)` turns the 2 learned features into
              3 class scores.
            """)
        }),
    ])
    return


@app.cell
def _(reset_button):
    import torch
    import torch.nn as nn

    # Referencing the reset button makes this cell re-run (new random weights) when it is clicked
    reset_button

    class MLP(nn.Module):
        """Three‑hidden‑layer perceptron for the 2‑D three‑class dataset."""
        def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, int, int] = (16, 16, 2), out_dim: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.ReLU(),
                nn.Linear(hidden_dims[2], out_dim)
            )

        def forward(self, x):
            return self.net(x)

    # Instantiate the model
    mlp_model = MLP()
    mlp_model
    return mlp_model, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Training

    At first the network's weights are random, so its guesses are random too. Training repeats a simple
    loop:

    1. **Guess** the class of a small batch of training points.
    2. **Measure** how wrong the guesses were. This error score is called the **loss**.
    3. **Nudge** every weight slightly in the direction that would have reduced the loss.

    One full pass through all the training points is called an **epoch**. We train for **20 epochs**.

    *Note: the network is trained once (20 epochs) when the notebook opens. Pressing **Train Model**
    continues training the same network from where it left off. Pressing **Reset Model** throws away
    everything learned and trains a fresh network, starting from a new set of random weights.*
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label='Train Model')
    reset_button = mo.ui.run_button(label='Reset Model', kind='warn')
    mo.hstack([run_button, reset_button], justify='start')
    return reset_button, run_button


@app.cell
def _(
    X_test,
    X_train,
    mlp_model,
    mo,
    nn,
    plt,
    run_button,
    torch,
    y_test,
    y_train,
):
    # Runs once when the notebook opens; pressing Train Model trains the same network for 20 more epochs
    run_button

    # Convert NumPy arrays to torch tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    # Data loader for mini‑batch training
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Loss, optimizer and training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
    epochs = 20
    _train_losses = []

    # Training loop
    mlp_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = mlp_model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        _train_losses.append(epoch_loss)

    # Plot learning curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), _train_losses, marker="o")
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Evaluation on test set
    from sklearn.metrics import precision_score, recall_score

    mlp_model.eval()
    with torch.no_grad():
        test_logits = mlp_model(X_test_t)
        test_preds = torch.argmax(test_logits, dim=1).numpy()
        test_accuracy = (test_preds == y_test).mean()
        test_precision = precision_score(y_test, test_preds, average="macro")
        test_recall = recall_score(y_test, test_preds, average="macro")

    # Display results together with the plot
    mo.vstack([
        plt.gca(),
        mo.md(
            f"**Test Accuracy:** {test_accuracy:.2%}<br>"
            f"**Test Precision:** {test_precision:.2%}<br>"
            f"**Test Recall:** {test_recall:.2%}"
        )
    ])
    return (test_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ### Reading the results

        The **learning curve** shows the loss after each epoch. A curve that drops and then flattens out
        means the network is learning and gradually settling on a good solution. If it stays high, the
        network got stuck; try pressing **Train Model** again, or **Reset Model** to start over from
        new random weights.

        The scores underneath are measured on the **test set**, the "exam" points the network never saw
        during training.
        """),
        mo.accordion({
            "Learn more: accuracy, precision and recall": mo.md(r"""
            - **Accuracy**: the fraction of test points that were given the correct class.
            - **Precision**: when the network says "this is class X", how often is it right?
              (averaged over the three classes)
            - **Recall**: of all the points that truly are class X, how many did the network find?
              (averaged over the three classes)

            With balanced classes like ours, all three numbers are usually close to each other.
            """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Looking inside the black box

    Now for the most interesting part. Remember the 2-neuron **bottleneck** layer? For every point in the
    dataset we record the two numbers that come out of that layer and plot them, keeping each point's
    original color.

    This shows the data **through the eyes of the network**: it is the re-drawn map that the final layer
    uses to make its decision.
    """)
    return


@app.cell
def _(X, mlp_model, nn, plt, test_accuracy, torch, y):
    test_accuracy  # wait until training has finished

    # Compute hidden representations from the last hidden layer (2‑D) for the entire dataset
    # Extract the sub‑network up to the last hidden ReLU (includes Linear → ReLU)
    _feature_extractor = nn.Sequential(*list(mlp_model.net.children())[:6])

    # Convert full dataset to torch tensor
    _X_full_t = torch.from_numpy(X).float()

    # Get 2‑D hidden features
    hidden_vals = _feature_extractor(_X_full_t).detach().cpu().numpy()

    # Plot the hidden space, colored by original class label
    plt.figure(figsize=(6, 5))
    scatter_hidden = plt.scatter(
        hidden_vals[:, 0],
        hidden_vals[:, 1],
        c=y,
        cmap="viridis",
        edgecolor="k"
    )
    plt.title("2‑D Hidden Layer Representation (Last Hidden Layer)")
    plt.xlabel("Hidden Dimension 1")
    plt.ylabel("Hidden Dimension 2")
    plt.grid(True)
    plt.legend(*scatter_hidden.legend_elements(), title="Class")

    # Return the matplotlib Axes object for marimo to render
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            **The key insight:** compare this plot with the original data. The moons and the ring have
            been pulled apart into separate regions, and now the colors **can** be separated with straight
            lines. That is exactly what the final layer of the network does: it is just a simple
            straight-line classifier. All the hard work was done by the hidden layers, which **learned
            features** that make the problem easy.
            """),
            kind="info",
        ),
        mo.md(r"""
        A few things you may notice:

        - Many points sit exactly on an axis or at the corner (0, 0). This is the ReLU step at work:
          any negative value is set to zero.
        - The picture looks different each time you press **Reset Model** and retrain. There are many
          ways to re-draw the map, and the random starting weights decide which one the network finds.
          Sometimes it lands on a poor one where the colors still overlap; if so, reset and try again.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - **Some problems cannot be solved with straight lines** in their original form.
    - **Hidden layers transform the data**, moving points into a new space (the learned **features**).
    - In that new space, **the final layer only needs to draw straight lines**. Neural networks learn the
      features and the classifier together, directly from examples.

    ### Things to try

    - Press **Train Model** several times and watch how the learning curve and the hidden-layer plot change.
    - Press **Reset Model** a few times and compare the hidden-layer plots that different random
      starting points produce.
    - Move the **Samples per class** slider to use more or fewer points, then retrain. Does less data
      make learning harder?
    - Compare the hidden-layer plot from a poorly trained network (high loss) with a well trained one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "References": mo.md(r"""
        - **Backpropagation:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
          representations by back-propagating errors. *Nature*, 323(6088), 533–536.
          [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0)
        - **ReLU:** Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann
          machines. *Proceedings of the 27th International Conference on Machine Learning (ICML 2010)*,
          807–814.
        - **Adam optimizer:** Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
          *ICLR 2015*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
        - **Book:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
          [deeplearningbook.org](https://www.deeplearningbook.org/)
        - **scikit-learn** (the two-moons data, the train/test split, precision and recall): Pedregosa, F., et al.
          (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12,
          2825–2830. [jmlr.org/papers/v12/pedregosa11a](https://jmlr.org/papers/v12/pedregosa11a.html)
        - **PyTorch** (the library used to build and train the networks): Paszke, A., et al. (2019). PyTorch:
          An imperative style, high-performance deep learning library. *Advances in Neural Information
          Processing Systems 32* (NeurIPS 2019). [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
        """)
    })
    return


if __name__ == "__main__":
    app.run()
