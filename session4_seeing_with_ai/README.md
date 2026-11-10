# Session 4 — Seeing with AI: Image Processing from Pixels to Deployment

How to go from a folder of raw microscope images to a model in an app, with less manual labeling. A foundation
model drafts the labels, experts correct them, and a small detector is trained, expanded with its own labels and
deployed. The example is counting nematodes under the microscope (the Nema-AI project of NC State and NCDA&CS).

- [`Slides.pdf`](Slides.pdf): slides for the session.
- [`HandsOn/`](HandsOn/): the workflow built during the session.

## Data

The hands-on uses a dataset for visual nematode detection:
[`nematode_yolo.zip`](https://drive.google.com/file/d/1cdMcoLGnJxZ4wyi4F4NM8O8auxCKUw3K/view?usp=sharing)
(Google Drive, about 52 MB). Download it and unzip it before the session.

## Parts

1. **The task and the data (15 min).** Nematodes under the microscope: why counting by hand is a bottleneck, and
   what the Nema-AI project learned (the bottleneck was annotation, not the model).
2. **Pre-label with SAM 2 (20 min).** A foundation model drafts the labels from a few prompts.
3. **Re-annotate in Labelbox (20 min).** Experts correct the drafts instead of drawing from scratch.
4. **Semi-supervised training (25 min).** Train a small detector, let it label the remaining images
   (pseudo-labels) and retrain.
5. **Evaluate (10 min).** Model counts vs. expert counts.
6. **Deploy with Streamlit (20 min).** From a model file to an app. Uncertain results from the app go back to
   Labelbox, and the model is retrained.
