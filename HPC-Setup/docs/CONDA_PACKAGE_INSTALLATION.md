# Installing Packages with Conda

## Installing scikit-learn

### Quick Install

```bash
conda install scikit-learn
```

### From Conda-Forge (alternative)

```bash
conda install -c conda-forge scikit-learn
```

### With pip (if conda has issues)

```bash
pip install scikit-learn
```

## Installation in Your GPU Notebook Setup

Since you're running the CNN notebook, install scikit-learn in your `ai-workshop` environment:

```bash
# On compute node
conda activate ai-workshop
conda install scikit-learn
```

Or add it to your setup script:

```bash
# In your Jupyter server startup, you can install before starting:
conda activate ai-workshop
conda install scikit-learn  # Add this line
/opt/Anaconda/bin/jupyter server ...
```

## Common Data Science Packages

Here are packages commonly used with PyTorch that you might need:

```bash
# Core data science
conda install numpy pandas matplotlib scikit-learn

# Jupyter and notebook support
conda install ipykernel jupyter jupyterlab

# GPU/ML
conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia

# All-in-one for this workshop
conda install numpy pandas matplotlib scikit-learn ipykernel pytorch pytorch-cuda=12.8 -c pytorch -c nvidia
```

## Installing Multiple Packages at Once

You can install several packages in one command:

```bash
conda install numpy pandas matplotlib scikit-learn
```

Or from your setup script, after activating the environment:

```bash
conda activate ai-workshop
conda install pytorch pytorch-cuda=12.8 scikit-learn matplotlib -c pytorch -c nvidia
```

## Verifying Installation

Check if scikit-learn is installed:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

## If Using the CNN Notebook

The CNN notebook requires:
- `torch` (PyTorch) - for neural networks
- `numpy` - for numerical computation
- `matplotlib` - for plotting
- `sklearn` (scikit-learn) - for confusion matrix display

Install all at once:

```bash
conda activate ai-workshop
conda install pytorch pytorch-cuda=12.8 numpy matplotlib scikit-learn -c pytorch -c nvidia
```

## Complete Updated Setup

Here's the full setup with all required packages:

```bash
# 1. Request GPU
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash

# 2. Load modules
module load Python/3.12.7 cuda/12.8 Anaconda/2025.06

# 3. Setup conda
conda init bash
source /opt/Anaconda/etc/profile.d/conda.sh
conda activate ai-workshop

# 4. Install all required packages
conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia
pip install matplotlib numpy scipy jupyterlab scikit-learn

# 5. Register kernel
conda install ipykernel
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"

# 6. Start Jupyter
/opt/Anaconda/bin/jupyter server --no-browser --ip=127.0.0.1 --port=8888
```

## Troubleshooting Package Installation

### Issue: "Package not found"

Try conda-forge:
```bash
conda install -c conda-forge scikit-learn
```

### Issue: "Conflicts with existing packages"

Check what's installed:
```bash
conda list
```

Update conda first:
```bash
conda update conda
conda install scikit-learn
```

### Issue: "Environment not activated"

Make sure you're in the right environment:
```bash
conda activate ai-workshop
conda install scikit-learn
```

Verify you're in the right environment (should show `(ai-workshop)` in prompt):
```bash
conda info | grep "active environment"
```
