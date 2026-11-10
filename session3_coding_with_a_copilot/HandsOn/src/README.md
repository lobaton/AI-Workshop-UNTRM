# Bean-leaf analysis and transfer learning

This folder contains a hands-on exploratory analysis and a CPU comparison of
pretrained image classifiers. The task is to classify bean leaves as
**angular leaf spot**, **bean rust** or **healthy**. The source data is Session
2's [`beans128.npz`](../../../session2_inside_the_black_box/src/data/beans128.npz);
its [data README](../../../session2_inside_the_black_box/src/data/README.md)
describes the dataset, format and provenance.

## Folder contents

| Item | Purpose |
| --- | --- |
| [`beans_eda.ipynb`](beans_eda.ipynb) | Training-image examples, class balance, dimensions, color and brightness analysis. |
| [`beans_transfer_learning.ipynb`](beans_transfer_learning.ipynb) | Explains the experiment and displays saved validation and test results. |
| [`beans_transfer.py`](beans_transfer.py) | CPU training, validation, final test evaluation and local-image prediction. |
| [`test_beans_transfer.py`](test_beans_transfer.py) | Focused tests for dataset handling, models, checkpoint reload and evaluation. |
| [`PROMPTS_LOG.md`](PROMPTS_LOG.md) | Main user prompts and follow-up selections from this session. |
| [`DECISIONS_LOG.md`](DECISIONS_LOG.md) | Dataset, training, selection and evaluation decisions. |
| [`beans_results_run1/`](beans_results_run1/) | Saved configuration, six checkpoints, validation histories and final results for the completed run. |
| `__pycache__/` | Automatically generated Python bytecode cache; not an experiment result. |

Within [`beans_results_run1/`](beans_results_run1/),
[`run_config.json`](beans_results_run1/run_config.json) records the data hash,
hyperparameters, preprocessing and software versions.
[`validation_results.json`](beans_results_run1/validation_results.json) records
all six comparisons and the selected checkpoint;
[`test_results.json`](beans_results_run1/test_results.json) records the final
evaluation. Each model and variant also has a `.pt` checkpoint and a `.json`
validation history. The six checkpoints occupy roughly 150 MB in total.

## Results

The original dataset splits were used: 1,034 training images, 133 validation
images and 128 test images. The table shows each model's best **validation**
checkpoint. Fine-tuning begins from the corresponding frozen checkpoint; the
individual stage times in the JSON reports do not include both stages.

| Pretrained backbone | Frozen accuracy | Fine-tuned accuracy | Frozen macro F1 | Fine-tuned macro F1 |
| --- | ---: | ---: | ---: | ---: |
| EfficientNet-B0 | 95.49% | 96.99% | 95.55% | 97.00% |
| ResNet-18 | 92.48% | **98.50%** | 92.45% | 98.50% |
| MobileNetV3-Large | 93.23% | 94.74% | 93.17% | 94.68% |

The training majority-class baseline reached 33.83% validation accuracy.
**Fine-tuned ResNet-18** was selected using validation accuracy and was the
only model evaluated on the test set:

- Test accuracy: **94.53%** (121/128); 95% Wilson interval: **89.14%–97.33%**.
- Test macro F1: **94.54%**.
- Test recall: angular leaf spot **100.00%**, bean rust **86.05%**, healthy
  **97.62%**.
- Six bean-rust examples were predicted as angular leaf spot; one healthy
  example was predicted as bean rust. See the
  [test report](beans_results_run1/test_results.json) for precision, recall and
  the complete confusion matrix.

These are results from one run and a small held-out test set. Different images,
locations, lighting and backgrounds may change performance. The three-class
model cannot identify all possible diseases and is not an expert diagnosis.

## Running and reusing

See the [HandsOn README](../README.md) for dependencies, dataset location,
training and test commands, and checkpoint prediction. To review the existing
run without retraining, open
[`beans_transfer_learning.ipynb`](beans_transfer_learning.ipynb) and leave
`RUN_TRAINING = False`. For a new run, choose a **new** output directory; the
training script will not overwrite an existing one. The selected checkpoint is
[`resnet18_fine_tuned.pt`](beans_results_run1/resnet18_fine_tuned.pt).
