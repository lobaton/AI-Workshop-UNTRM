# Prompts log: bean-leaf classification

This is a concise record of the main user requests and follow-up choices in the
conversation that produced the bean-leaf exploratory analysis and model
comparison. Quoted passages are user messages; choices in forms are summarized
separately. This is not a complete transcript or a record of prompts used to
pretrain the models.

## 1. Explore examples and organize code

The user asked for examples from the Session 2 dataset:

> Look at session 2 folder and the file src/data/beans28.npz , which contains images from leafs that are health and with some diseases. You can look at the README for more details as well. Can you extract 6 images from each class and put them in a subfolder in the session 2 data directory.

The dataset documented and used in the session was `beans128.npz`. Six examples
per class were extracted from it. The user then set the location for subsequent
generated code:

> Please make sure to include all your code that you generate from this point on under a src directory in session 3's "HandsOn" directory.

## 2. Plan exploratory data analysis

> My goal is to create a model that predicts the different types of leaves that we have in the beans128.npz dataset from section 2. Before that, I wan to start with an exploratory data analysis. Can you give me a suggestion for a plan for this? What analysis can we do?

In the follow-up form, the user chose an interactive notebook, a practical EDA,
and detailed exploration of training images only, with validation and test
metadata used for counts. The resulting analysis is in
[`beans_eda.ipynb`](beans_eda.ipynb).

## 3. Consider pretrained models

> I am thinking about using a pre-trained model for this classification task of leaves. Could you give me some ideas of what I could use for that?

The user selected CPU execution, all available training images and accuracy as
the main priority. After discussing model options, the user initially selected
EfficientNet-B0 as the first model to focus on.

## 4. Expand and approve the comparison

The user refined the scope:

> I like everything that you mentioned. I do want to compare against ResNet-18, MobileNetV3-Large, but not BioCLIP. Also, I want to compare in all cases what happens when the features are frozen vs. when you fine-tune. Use the validation set for comparing models, and then report the results of the best model using the test set. Can you come up with a full plan for training and validation based on this?

In the follow-up form, the user chose validation **accuracy** as the primary
selection metric. The plan was approved for implementation. It produced six
experiments, the [comparison notebook](beans_transfer_learning.ipynb), and the
saved [validation](beans_results_run1/validation_results.json) and
[test](beans_results_run1/test_results.json) results.

## 5. Document the work

The user subsequently requested a prompts log, a decision log, and a README
summarizing the source folder and results. These documents describe the completed
run; they do not represent additional training experiments.

See [DECISIONS_LOG.md](DECISIONS_LOG.md) for the resulting technical decisions
and [README.md](README.md) for the folder overview and results.
