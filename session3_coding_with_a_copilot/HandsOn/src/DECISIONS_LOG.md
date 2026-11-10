# Decision log: bean-leaf classification

This log records the principal choices made during the conversation and the
implemented experiment. User-specified choices are identified below; other
technical settings are implementation choices recorded in
[`run_config.json`](beans_results_run1/run_config.json) and
[`beans_transfer.py`](beans_transfer.py).

## Data and scope

1. **Dataset and labels.** Use Session 2's
   [`beans128.npz`](../../../session2_inside_the_black_box/src/data/beans128.npz):
   128×128 RGB images labeled angular leaf spot, bean rust or healthy. These are
   health categories of bean leaves, not different leaf species.
2. **Original splits.** Use all 1,034 training images, 133 validation images
   for model selection and 128 test images for the final selected model. The
   user chose the full training split rather than Session 3's optional
   25-images-per-class exercise. Preserve the dataset's original splits.
3. **Exploratory analysis.** The user chose a practical interactive notebook.
   The EDA inspects training images and uses validation/test metadata only for
   class counts; it does not train a model.

## Models and training

4. **CPU and model choice.** The user chose CPU and an accuracy-oriented
   comparison of ImageNet-pretrained EfficientNet-B0, ResNet-18 and
   MobileNetV3-Large; BioCLIP was excluded. Compare each backbone with frozen
   features and with its final feature block fine-tuned, for six runs.
5. **Architecture and fine-tuning.** Replace the final classifier layer with a
   three-class linear layer. The fine-tuned variant starts from its corresponding
   best frozen checkpoint, unfreezes ResNet-18's `layer4` or the final
   `features` block of the other backbones, and trains that block and the head.
   Batch-normalization parameters and running statistics remain frozen.
6. **Preprocessing.** Use each torchvision weights version's documented
   inference transforms for validation and test. Training additionally uses
   random horizontal flips, rotations up to 10 degrees and modest brightness
   changes. Upsampling 128×128 images to model input size adds no new detail;
   the transforms also center-crop the resized images.
7. **Run settings.** Seed 42, AdamW, batch size 16, four CPU threads, at most
   20 epochs and early-stopping patience 5. Learning rates are 0.001 for the
   head and 0.0001 for the unfrozen backbone; weight decay is 0.01. The fine-tuned
   run follows the frozen run, so their reported elapsed times are separate
   stages rather than the total cost of producing a fine-tuned checkpoint.

## Selection and evaluation

8. **Checkpoint and model selection.** Within each run, select the checkpoint
   by validation accuracy, breaking ties with lower validation loss. Choose among
   the six using validation accuracy, then macro F1, then prefer frozen features,
   then the fixed model order EfficientNet-B0, ResNet-18, MobileNetV3-Large.
   The majority-class baseline is derived from training labels.
9. **Test only the winner.** Record the validation winner before evaluating its
   saved checkpoint on the held-out test split. Do not retrain on validation or
   compare other models using test scores. Report accuracy, macro F1, per-class
   precision and recall, confusion matrix and a 95% Wilson interval for accuracy.
10. **Result.** Fine-tuned ResNet-18 was selected with 98.50% validation accuracy.
    Its test accuracy was 94.53% (121/128) and macro F1 was 94.54%. The
    [validation report](beans_results_run1/validation_results.json) and
    [test report](beans_results_run1/test_results.json) are the detailed source
    of these results.

## Limitations and handoff

- Validation and test splits are small; the accuracy interval does not account
  for correlated images, dataset leakage or changes in field conditions.
- Backgrounds, lighting and potential near-duplicates can affect results.
  The model always selects one of three known classes and is not a substitute
  for expert disease diagnosis.
- Generated code lives under Session 3's `HandsOn/src/`. The
  [README](README.md) describes the files, how to reuse the checkpoint and how
  to run the experiment again.
