# GPU Jupyter Notebook Setup - Quick Reference

This folder contains complete documentation and scripts for running Jupyter notebooks with GPU support on the HPC cluster.

## Files in This Directory

### 📋 **GPU_NOTEBOOK_SETUP.md** (Main Documentation)
The comprehensive guide covering:
- Step-by-step setup instructions (6 steps)
- Complete workflow for 3 terminals
- GPU verification code
- Troubleshooting (including conda issues)
- Ready-to-use quick start script

**Start here!** This is your main reference document.

### 🔧 **CONDA_ACTIVATION_FIX.md** (Conda Troubleshooting)
Detailed solutions for conda activation issues:
- Why `conda init` appears to do nothing
- 4 different solution approaches
- Complete working sequence
- How to verify conda is active

Use this if you encounter conda activation problems.

## Quick Start (3 Steps)

1. **Allocate GPU resources** on login node:
   ```bash
   srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
   ```

2. **On compute node**, load modules and start Jupyter:
   ```bash
   module load Python/3.12.7 cuda/12.8 Anaconda/2025.06
   conda init bash
   source /opt/Anaconda/etc/profile.d/conda.sh
   conda create -n ai-workshop python=3.12.7
   conda activate ai-workshop
   conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia
   pip install matplotlib numpy scipy jupyterlab
   /opt/Anaconda/bin/jupyter server --no-browser --ip=127.0.0.1 --port=8888
   ```
   Note your SLURM Job ID from the prompt.

3. **In separate terminals** (see GPU_NOTEBOOK_SETUP.md for details):
   - Terminal 2: Find job ID with `squeue -u "$USER"`, then set up port forwarding
   - Terminal 3: SSH port forward from local machine (`ssh -L 8889:127.0.0.1:8889 user@login-node`)
   - In VS Code: Select "Existing Jupyter Server", paste: `http://127.0.0.1:8889/?token=YOUR_TOKEN_HERE`

## Common Issues

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Run: `source /opt/Anaconda/etc/profile.d/conda.sh` |
| `conda init` does nothing | Run the `source` command above immediately after |
| GPU not recognized | Verify you loaded `cuda/12.8` module |
| Can't connect to Jupyter | Check port forwarding is running and SLURM job ID is correct |
| VS Code can't connect | Ensure SSH tunnel is active, use port 8889, check full URL with token |

## VS Code Jupyter Connection with GitHub Copilot

To connect VS Code directly to the remote GPU kernel:

1. **Keep SSH port forwarding running** in a terminal:
   ```bash
   ssh -L 8889:127.0.0.1:8889 user@login-node
   ```
2. **In VS Code**, open a notebook file (`.ipynb`)
3. **Click the kernel selector** (top-right of notebook)
4. **Select "Existing Jupyter Server"**
5. **Paste the full URL** with port 8889 and token from Jupyter output:
   ```
   http://127.0.0.1:8889/?token=YOUR_TOKEN_HERE
   ```
6. **Press Enter** to connect

VS Code will show the compute node name (e.g., `g001`) in the kernel indicator. You can now use GitHub Copilot chat with GPU-accelerated code execution.

## Key Modules

- **Python/3.12.7** - Base Python
- **cuda/12.8** - CUDA toolkit for GPU
- **Anaconda/2025.06** - Conda package manager

## Key Commands

```bash
# Find your SLURM job ID (run on login node)
squeue -u "$USER"

# Check SLURM job details
squeue --me

# Verify GPU access in notebook
nvidia-smi -L

# See active conda environment
conda info

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

1. Read **GPU_NOTEBOOK_SETUP.md** for the full workflow
2. Use the Quick Start Script (`start_jupyter_gpu.sh`) for repeat sessions
3. Bookmark this folder for future reference

## Support

- For conda issues: See CONDA_ACTIVATION_FIX.md
- For connection issues: Check GPU_NOTEBOOK_SETUP.md Troubleshooting section
- GPU verification code: In GPU_NOTEBOOK_SETUP.md under "Verification"
