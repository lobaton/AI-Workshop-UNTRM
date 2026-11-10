# Session 3 — Adapting Pre-Trained Models

For a complete source-folder overview, experiment results, and the session's
prompt and decision logs, see [`src/README.md`](src/README.md).

## Bean-leaf exploratory data analysis

Open [`src/beans_eda.ipynb`](src/beans_eda.ipynb) in Jupyter or VS Code and run
its cells in order. It explores the training images in Session 2's
[`beans128.npz`](../../session2_inside_the_black_box/src/data/beans128.npz);
validation and test data are used only for label counts. The notebook requires
Python, Jupyter, NumPy, pandas, Matplotlib, and Pillow; it does not require a GPU.

Start Jupyter within this repository so the notebook can find the dataset. If
the dataset is stored elsewhere, set `BEANS_DATASET_PATH` to its full path before
starting Jupyter. The notebook describes the analyses and the next steps toward
a classifier; it does not train a model.

## Transfer learning: compare six classifiers

[`src/beans_transfer_learning.ipynb`](src/beans_transfer_learning.ipynb) explains
and displays the comparison; [`src/beans_transfer.py`](src/beans_transfer.py)
implements it. The experiments use ImageNet-pretrained EfficientNet-B0,
ResNet-18 and MobileNetV3-Large, each with a frozen backbone and with the final
feature block fine-tuned. All run on CPU.

Install compatible versions of PyTorch and torchvision, NumPy, Pillow,
scikit-learn, pandas and Jupyter in your chosen Python environment. The first
training run downloads pretrained torchvision weights if they are not cached.
From the repository root, for example:

```bash
python session3_coding_with_a_copilot/HandsOn/src/beans_transfer.py train \
  --output session3_coding_with_a_copilot/HandsOn/src/beans_results_run2
```

The output directory must not already exist. Use `--dataset /path/to/beans128.npz`
if the data is not at its usual Session 2 location. Use `--help` to see options,
including CPU threads, epochs, batch size and seed. Defaults are 20 epochs
maximum, patience 5, batch size 16, four CPU threads and seed 42. A complete
run may take substantial CPU time. In the notebook, set `RUN_TRAINING = True`
and use a new output directory to start a run, or leave it `False` to inspect an
existing completed run.

The program saves each checkpoint and validation history, `run_config.json`,
`validation_results.json` and `test_results.json` in the output directory.
The six variants use all original training images, with training-only
augmentation. The highest validation accuracy selects the winner; ties use
macro F1, then prefer frozen features and a fixed model order. Only that selected
checkpoint is evaluated on the original test split. No validation images are
used for fitting. The validation and test splits are small, and a model can
pick up backgrounds or perform poorly on images from different settings.

To classify another image using the checkpoint named in the validation report:

```bash
python session3_coding_with_a_copilot/HandsOn/src/beans_transfer.py predict \
  --checkpoint /path/to/selected_checkpoint.pt --image /path/to/leaf.jpg
```

The model always chooses among its three known classes; it cannot detect every
possible disease or replace an expert diagnosis. Before redistributing dataset
images or pretrained weights, check their respective licenses.

Run the focused checks from the source directory:

```bash
cd session3_coding_with_a_copilot/HandsOn/src
python -m unittest -v test_beans_transfer
```
