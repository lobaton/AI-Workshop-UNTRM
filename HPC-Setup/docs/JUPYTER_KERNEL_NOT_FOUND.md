# Troubleshooting: Conda Kernel Not Showing in VS Code

## Problem

You can connect to the Jupyter server in VS Code, but the `ai-workshop` conda kernel doesn't appear in the kernel selector.

## Root Cause

Jupyter needs to register the conda environment as a kernel. The environment exists, but Jupyter doesn't know about it yet.

## Solution

You need to install `ipykernel` in your conda environment and register it with Jupyter.

### Step 1: Activate Your Environment (On Compute Node)

```bash
conda activate ai-workshop
```

### Step 2: Install IPykernel

```bash
conda install ipykernel
```

Or using pip:

```bash
pip install ipykernel
```

### Step 3: Register the Kernel with Jupyter

```bash
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"
```

**Explanation of flags:**
- `--user`: Install for current user (not system-wide)
- `--name ai-workshop`: Internal kernel name
- `--display-name "AI Workshop (GPU)"`: Name shown in VS Code kernel selector

### Step 4: Restart Jupyter Server

Stop the Jupyter server (Ctrl+C in Terminal 1) and restart it:

```bash
/opt/Anaconda/bin/jupyter server \
  --no-browser \
  --ip=127.0.0.1 \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$HOME"
```

### Step 5: Reconnect in VS Code

1. In VS Code, click the kernel selector (top-right)
2. Select "Existing Jupyter Server" again
3. Paste your token URL: `http://127.0.0.1:8889/?token=YOUR_TOKEN`
4. Now you should see **"AI Workshop (GPU)"** in the kernel list

## Verify the Kernel is Registered

```bash
jupyter kernelspec list
```

Should show:
```
Available kernels:
  ai-workshop    /home/edgar.lobaton/.local/share/jupyter/kernels/ai-workshop
  python3        /opt/Anaconda/share/jupyter/kernels/python3
```

## Complete Setup Sequence

If you want to do the full setup from scratch:

```bash
# 1. Allocate GPU
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash

# 2. Load modules
module load Python/3.12.7
module load cuda/12.8
module load Anaconda/2025.06

# 3. Initialize and activate conda
conda init bash
source /opt/Anaconda/etc/profile.d/conda.sh
conda activate ai-workshop

# 4. Install ipykernel (do this only once)
conda install ipykernel

# 5. Register kernel with Jupyter (do this only once)
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"

# 6. Start Jupyter server
/opt/Anaconda/bin/jupyter server \
  --no-browser \
  --ip=127.0.0.1 \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$HOME"
```

## If Still Not Showing

Try these additional troubleshooting steps:

### 1. Clear VS Code's kernel cache

```bash
# On your local machine
rm -rf ~/.cache/vscode-server/extensions/ms-python.python/*/kernels/
```

Or in VS Code:
- Open Command Palette (Ctrl+Shift+P)
- Type: "Developer: Reload Window"
- Press Enter

### 2. Manually specify kernel in notebook

In your notebook's top-right, click kernel selector and type:
```
ai-workshop
```

### 3. Check Jupyter can see the kernel

On the compute node:
```bash
jupyter kernelspec list
python -m jupyter kernelspec list
```

### 4. Reinstall ipykernel

```bash
conda remove ipykernel
conda install ipykernel
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)" --force
```

## Quick Checklist

- [ ] Activated `ai-workshop` conda environment
- [ ] Ran `conda install ipykernel`
- [ ] Ran `python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"`
- [ ] Restarted Jupyter server
- [ ] Reconnected in VS Code
- [ ] Verified with `jupyter kernelspec list`

Once ipykernel is registered, the kernel should appear immediately in VS Code.
