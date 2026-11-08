# GPU Notebook Setup Documentation

This document describes how to set up and run Jupyter notebooks with GPU support on the HPC cluster.

**📌 Quick Start**: See [SETUP_README.md](./SETUP_README.md) for a quick reference guide and overview of all documentation.

## Overview

The setup involves six key steps:
1. **SLURM allocation**: Request GPU resources from the cluster
2. **Module loading**: Load CUDA, Python, and conda environment
3. **Conda environment**: Create and activate your Python environment
4. **Jupyter server**: Start the notebook server on the compute node
5. **Port forwarding**: Create a secure tunnel from login node to compute node
6. **Local access**: Connect to Jupyter from your local machine

## Prerequisites

- Access to an HPC cluster with GPU nodes
- SSH access to the login node
- Sufficient allocation time (cluster dependent)

## Step 1: Allocate GPU Resources with SLURM

On the **login node**, start an interactive SLURM job with GPU access:

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
```

### SLURM Command Breakdown

- **--partition=gpu**: Request resources from the GPU partition
- **--gres=gpu:1**: Request 1 GPU (generic resource)
- **--time=01:00:00**: Allocate for 1 hour (HH:MM:SS format)
- **--pty bash**: Start an interactive bash shell

**Note**: Adjust `--time` based on your needs. The command will return a prompt on an allocated compute node once resources are available.

Once this completes, you'll be on a compute node (e.g., `g001`) with GPU access. Your SLURM job ID will be stored in `$SLURM_JOB_ID`.

## Step 2: Load Required Modules

On the **compute node** (or within your SLURM job), load the necessary modules in this order:

```bash
module load Python/3.12.7
module load cuda/12.8
module load Anaconda/2025.06
```

### Module Explanation

- **Python/3.12.7**: Base Python environment
- **cuda/12.8**: CUDA toolkit for GPU support (required for PyTorch and other GPU-accelerated libraries)
- **Anaconda/2025.06**: Conda package manager for environment management

## Step 3: Create and Activate a Conda Environment

First, initialize conda in your current bash shell:

```bash
conda init bash
source /opt/Anaconda/etc/profile.d/conda.sh
```

**Note**: The `conda init` command modifies your `~/.bashrc` file, but you need to source the initialization immediately to use conda in the current session.

Create a dedicated conda environment for your work:

```bash
conda create -n ai-workshop python=3.12.7
conda activate ai-workshop
```

If `conda activate` doesn't work, try:
```bash
source activate ai-workshop
```

### Install Required Packages

Install PyTorch and other dependencies:

```bash
# PyTorch with CUDA 12.8 support
conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia

# Other useful packages (including scikit-learn for the CNN notebook)
pip install matplotlib numpy scipy jupyterlab scikit-learn

# IMPORTANT: Install ipykernel so Jupyter recognizes this environment
conda install ipykernel

# Register the kernel for VS Code
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"
```

## Step 4: Start Jupyter Server on Compute Node

On the **compute node** (within your allocation), start the Jupyter server:

```bash
/opt/Anaconda/bin/jupyter server \
  --no-browser \
  --ip=127.0.0.1 \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$HOME"
```

**Important**: The server binds to `127.0.0.1` (localhost) for security. You'll need port forwarding to access it.

## Step 5: Set Up Port Forwarding (Login Node)

First, find your SLURM job ID on the **login node**:

```bash
squeue -u "$USER"
```

This will show your running jobs. Note the job ID in the first column.

Run this command on the **login node** in a separate terminal. Replace `31983` with your actual SLURM job ID from the command above:

```bash
ncat -l -k 127.0.0.1 8889 \
  --sh-exec 'srun --quiet --unbuffered --jobid=31983 --overlap --ntasks=1 --gres=none sh -c "stty raw -echo <&1 || exit 1; exec ncat 127.0.0.1 8888"'
```

### Port Forwarding Explanation

- **ncat -l -k 127.0.0.1 8889**: Listen on port 8889 on the login node
- **--sh-exec**: Execute a shell command for each incoming connection
- **srun --jobid=31983**: Route connections back to your compute job
- **--ntasks=1 --gres=none**: Use minimal resources for the forwarding tunnel
- **ncat 127.0.0.1 8888**: Connect to Jupyter server on the compute node

## Step 6: Access Jupyter from VS Code

VS Code can connect directly to the remote Jupyter server as a kernel for GPU-accelerated notebook execution with GitHub Copilot integration in chat.

1. **In VS Code**, open a notebook file (`.ipynb`)
2. **Click the kernel selector** in the top-right corner (shows current kernel)
3. **Select "Existing Jupyter Server"**
4. **Paste the full Jupyter server URL** with the port changed to 8889:
   ```
   http://127.0.0.1:8889/?token=<YOUR_TOKEN_HERE>
   ```
   
   Example (using your Jupyter server output with port 8889):
   ```
   http://127.0.0.1:8889/?token=f5e5c73460b28b9be70ad4a136303de39661f9cbedfb1e19
   ```

5. **Press Enter** to connect

VS Code will now use the remote GPU kernel for all notebook cells. You'll see the compute node name (e.g., `g001`) in the kernel indicator once connected.

**Important**: 
- The SSH port forwarding (Terminal 3) must be active: `ssh -L 8889:127.0.0.1:8889 user@login-node`
- Use port **8889** (not 8888) for the VS Code connection
- You can now use GitHub Copilot chat while working with GPU-accelerated code

## Verification

To verify GPU access in your notebook, run this code cell:

```python
import os
import shutil
import socket
import subprocess

print(f"Kernel host: {socket.gethostname()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi is None:
    print("nvidia-smi is not on this kernel's PATH. GPU availability is unknown.")
else:
    result = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        print(f"nvidia-smi failed with exit code {result.returncode}.")
    elif not result.stdout.strip():
        print("nvidia-smi did not report any GPUs.")
```

Expected output should show:
- Compute node hostname (e.g., `g001`)
- GPU index (e.g., `CUDA_VISIBLE_DEVICES: 0`)
- Available GPU(s) from `nvidia-smi` output

## Troubleshooting

### Issue: "conda activate not found" or "conda: command not found"

**Cause**: The conda initialization hasn't been loaded in your current shell session.

**Solution**: Manually source the conda initialization:

```bash
source /opt/Anaconda/etc/profile.d/conda.sh
conda activate ai-workshop
```

### Issue: "conda init" runs but nothing happens

**Cause**: `conda init` modifies `~/.bashrc`, but the change only takes effect in new shells. Your current shell doesn't have conda initialized yet.

**Solution**: Initialize conda in your current session immediately after `conda init`:

```bash
conda init bash
source /opt/Anaconda/etc/profile.d/conda.sh
conda activate ai-workshop
```

### Issue: "Kernel not found" or "No kernels available" in VS Code

**Symptoms**: Connected to Jupyter server successfully, but the `ai-workshop` kernel doesn't appear in the kernel selector

**Cause**: The conda environment needs to be registered as a Jupyter kernel using `ipykernel`

**Solution** (on the compute node):
1. Activate your environment:
   ```bash
   conda activate ai-workshop
   ```
2. Install ipykernel:
   ```bash
   conda install ipykernel
   ```
3. Register the kernel:
   ```bash
   python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"
   ```
4. Restart the Jupyter server (Ctrl+C and restart)
5. Reconnect in VS Code

To verify the kernel was registered:
```bash
jupyter kernelspec list
```

Should show `ai-workshop` in the list.

**See also**: [JUPYTER_KERNEL_NOT_FOUND.md](./JUPYTER_KERNEL_NOT_FOUND.md) for detailed troubleshooting

### Issue: "ncat command not found"
**Solution**: `ncat` is usually available as part of `nmap` or netcat. Try `nc` instead or load the appropriate module.

### Issue: VS Code can't connect to Jupyter server

**Symptoms**: VS Code shows "Connection refused" or "Can't reach server"

**Cause**: The SSH port forwarding tunnel is not active or the URL is incorrect.

**Solution**:
1. Ensure SSH port forwarding is running in a terminal:
   ```bash
   ssh -L 8889:127.0.0.1:8889 user@login-node.cluster.edu
   ```
2. Copy the **full URL with token** from Terminal 1 Jupyter output
3. In VS Code, paste exactly: `http://127.0.0.1:8889/?token=YOUR_TOKEN_HERE`
4. Press Enter and wait a few seconds for connection

### Issue: VS Code kernel stuck or slow

**Cause**: Remote computation on GPU is slow for first cell or kernel needs initialization

**Solution**:
- Wait longer for first cell to execute (GPU allocation may take seconds)
- Run the GPU verification code to confirm kernel is working
- Check Terminal 1 for Jupyter server messages/errors

### Issue: "Can't connect to Jupyter server"
**Solution**: 
- Verify your SLURM job ID is correct
- Check that the Jupyter server is still running on the compute node
- Ensure the port forwarding command is still active

### Issue: "CUDA_VISIBLE_DEVICES: (not set)"
**Solution**: 
- Verify you loaded the `cuda/12.8` module
- Check that your SLURM job requested GPU resources (`--gres=gpu:1`)

### Issue: PyTorch not recognizing GPU
**Solution**:
- Verify CUDA is installed: `nvcc --version`
- Reinstall PyTorch with correct CUDA version:
  ```bash
  conda remove pytorch pytorch-cuda
  conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia
  ```

## Complete Setup Workflow

Here's the complete workflow from start to finish:

### Terminal 1: Request GPU and Start Jupyter

```bash
# On login node: Request GPU allocation
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash

# You are now on a compute node (e.g., g001)
# Load modules
module load Python/3.12.7
module load cuda/12.8
module load Anaconda/2025.06

# Initialize conda in current shell
conda init bash
source /opt/Anaconda/etc/profile.d/conda.sh

# Create environment (first time only)
conda create -n ai-workshop python=3.12.7
conda activate ai-workshop

# Install packages (first time only)
conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia
pip install matplotlib numpy scipy jupyterlab scikit-learn

# IMPORTANT: Install ipykernel and register kernel for VS Code (first time only)
conda install ipykernel
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"

# Start Jupyter server
/opt/Anaconda/bin/jupyter server \
  --no-browser \
  --ip=127.0.0.1 \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$HOME"

# IMPORTANT: Copy the full token URL shown in output, e.g.:
# http://127.0.0.1:8888/?token=f5e5c73460b28b9be70ad4a136303de39661f9cbedfb1e19
```

### Terminal 2: Set Up Port Forwarding

```bash
# On login node in a NEW terminal: Get your job ID
squeue -u "$USER"

# Look for your job in the output, note the JOBID from the first column
# Set up port forwarding (replace 31983 with your actual job ID from squeue)
ncat -l -k 127.0.0.1 8889 \
  --sh-exec 'srun --quiet --unbuffered --jobid=31983 --overlap --ntasks=1 --gres=none sh -c "stty raw -echo <&1 || exit 1; exec ncat 127.0.0.1 8888"'

# Leave this terminal running
```

### VS Code Connection

**This is the primary method to access your GPU notebook with GitHub Copilot:**

1. **In VS Code**, open a notebook file (`.ipynb`)
2. **Click the kernel selector** in the top-right corner
3. **Select "Existing Jupyter Server"**
4. **Paste the token URL** from Terminal 1 output, but **change port 8888 to 8889**:
   ```
   http://127.0.0.1:8889/?token=f5e5c73460b28b9be70ad4a136303de39661f9cbedfb1e19
   ```
5. **Press Enter** to connect

VS Code will show the compute node name (e.g., `g001`) in the kernel indicator. You can now use GitHub Copilot chat with GPU-accelerated code execution.

## Quick Start Script

Save this as `start_jupyter_gpu.sh` on the compute node for easier future use:

```bash
#!/bin/bash
set -e

echo "=== GPU Jupyter Setup ==="
echo ""
echo "Loading modules..."
module load Python/3.12.7
module load cuda/12.8
module load Anaconda/2025.06

echo "Initializing conda..."
conda init bash
source /opt/Anaconda/etc/profile.d/conda.sh

echo "Activating conda environment..."
conda activate ai-workshop

echo ""
echo "=== Jupyter Server Starting ==="
echo "Compute node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "=== PORT FORWARDING SETUP ==="
echo "Run this command in a NEW terminal on the login node:"
echo ""
echo "ncat -l -k 127.0.0.1 8889 --sh-exec 'srun --quiet --unbuffered --jobid=$SLURM_JOB_ID --overlap --ntasks=1 --gres=none sh -c \"stty raw -echo <&1 || exit 1; exec ncat 127.0.0.1 8888\"'"
echo ""
echo "Then SSH from your local machine and keep the tunnel active:"
echo "ssh -L 8889:127.0.0.1:8889 user@login-node"
echo ""
echo "=== VS CODE CONNECTION ==="
echo "In VS Code:"
echo "1. Click kernel selector (top-right)"
echo "2. Select 'Existing Jupyter Server'"
echo "3. Paste: http://127.0.0.1:8889/?token=<TOKEN_FROM_BELOW>"
echo ""
echo "=== Starting Jupyter Server ==="
echo ""

/opt/Anaconda/bin/jupyter server \
  --no-browser \
  --ip=127.0.0.1 \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$HOME"
```

Make it executable:
```bash
chmod +x start_jupyter_gpu.sh
```

Then use it:
```bash
# After SLURM allocation
./start_jupyter_gpu.sh
```

## References

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Jupyter Server Documentation](https://jupyter-server.readthedocs.io/)
- [SLURM Job Management](https://slurm.schedmd.com/)
