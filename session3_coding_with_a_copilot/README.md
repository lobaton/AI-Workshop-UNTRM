# Session 3 — Coding with a Copilot: Adapting Pre-Trained Models for Your Research

How to get good results when you have only a few labeled examples. You reuse models that others trained on large
datasets, and you write the code with the help of an AI code assistant. The examples use bean-leaf photos.

- [`Slides.pdf`](Slides.pdf): slides for the session.
- [`HandsOn/`](HandsOn/): the code written with the copilot during Part 3.

## Parts

1. **Reusing models when data is scarce (40 min).**
   - Transfer learning: feature extraction vs. fine-tuning, and habits for fine-tuning.
   - Foundation models (CLIP, DINOv2, BioCLIP, ESM-2, SAM) and how they learn without labels.
   - Vision-language models (VLMs): asking questions about an image in words.
   - Data augmentation, generative augmentation and regularization.
   - A recipe for small datasets.
2. **AI code assistants (20 min).** Getting started with GitHub Copilot.
3. **Use cases by prompting (60 min).** The experiment ladder on bean leaves, with only 25 leaves per class and
   the same 128 test images at every step:
   1. Prompt a VLM with the three labels (no training, no code).
   2. CLIP zero-shot (no training images at all).
   3. A small CNN trained from scratch.
   4. A frozen pre-trained backbone plus a linear classifier.
   5. Fine-tuning the top layers, with augmentation.
   6. Optional: regularization (dropout, weight decay) and model tuning.

   Which step gives the best accuracy for the effort?

The data is the Makerere AI Lab "beans" dataset at 128×128 pixels:
[`beans128.npz`](../session2_inside_the_black_box/src/data/beans128.npz), the same file used in Session 2. Its
format is described in [`data/README.md`](../session2_inside_the_black_box/src/data/README.md).
