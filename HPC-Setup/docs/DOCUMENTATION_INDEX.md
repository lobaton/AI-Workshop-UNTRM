# GPU Jupyter Notebook Setup - Complete Documentation Index

All documentation for running Jupyter notebooks with GPU support on HPC cluster and accessing them via VS Code with GitHub Copilot.

## 📋 Main Documentation

### [GPU_NOTEBOOK_SETUP.md](./GPU_NOTEBOOK_SETUP.md) - **START HERE**
Complete 6-step guide for setting up GPU-accelerated Jupyter notebooks with VS Code.
- SLURM GPU allocation
- Module loading (Python, CUDA, Anaconda)
- Conda environment creation with PyTorch
- Jupyter server startup
- Port forwarding setup
- VS Code kernel connection
- GPU verification code
- Comprehensive troubleshooting

**Best for**: First-time setup, complete workflow reference

---

## 🔧 Troubleshooting & Setup Guides

### [JUPYTER_KERNEL_NOT_FOUND.md](./JUPYTER_KERNEL_NOT_FOUND.md)
Dedicated guide for when conda kernel doesn't appear in VS Code.
- Root cause explanation
- 5-step solution with ipykernel
- Multiple troubleshooting approaches
- Quick verification checklist

**Best for**: "My kernel is missing!" problems

### [CONDA_ACTIVATION_FIX.md](./CONDA_ACTIVATION_FIX.md)
Solutions for conda activation issues and environment initialization.
- Why `conda init` appears to do nothing
- 4 different solution approaches
- Manual shell initialization
- Complete working sequence

**Best for**: Conda environment activation problems

### [CONDA_PACKAGE_INSTALLATION.md](./CONDA_PACKAGE_INSTALLATION.md)
Guide for installing Python packages with conda.
- Quick scikit-learn installation
- Common data science packages
- Multiple package installation
- Complete setup workflow
- Installation troubleshooting

**Best for**: Installing packages (scikit-learn, pytorch, jupyter, etc.)

---

## 📚 Reference Guides

### [SETUP_README.md](./SETUP_README.md)
Quick reference with 3-step quick start and common issues table.
- File overview and navigation
- 3-step quick start workflow
- Common issues quick-fix table
- Key modules and commands
- VS Code connection with GitHub Copilot

**Best for**: Quick lookup and first-time orientation

### [TEST_GPU_PYTORCH.md](./TEST_GPU_PYTORCH.md)
Guide to testing GPU functionality with the CNN notebook.
- CNN notebook location and features
- GPU verification code samples
- Expected output (GPU vs CPU)
- Performance expectations
- Links to other GPU notebooks

**Best for**: Testing GPU is working correctly

### [MATPLOTLIB_JUPYTER_PLOTTING.md](./MATPLOTLIB_JUPYTER_PLOTTING.md)
Guide for using matplotlib in Jupyter notebooks on remote GPU server.
- Magic command setup (`%matplotlib inline`)
- Common plotting issues and solutions
- Best practices for remote plotting
- CNN notebook plotting examples
- Troubleshooting remote plot display

**Best for**: Making plots display correctly in notebooks

---

## 🚀 Quick Start Paths

### I'm new to this setup
1. Read: [SETUP_README.md](./SETUP_README.md)
2. Follow: [GPU_NOTEBOOK_SETUP.md](./GPU_NOTEBOOK_SETUP.md) (Section: Complete Setup Workflow)
3. Test: [TEST_GPU_PYTORCH.md](./TEST_GPU_PYTORCH.md)

### I'm stuck on something
1. Check: [SETUP_README.md](./SETUP_README.md) Common Issues table
2. Search: Use Ctrl+F to find your issue in [GPU_NOTEBOOK_SETUP.md](./GPU_NOTEBOOK_SETUP.md) Troubleshooting
3. Drill down: Use specific guide (JUPYTER_KERNEL_NOT_FOUND, CONDA_ACTIVATION_FIX, etc.)

### I need to do a task
- **Install a package**: [CONDA_PACKAGE_INSTALLATION.md](./CONDA_PACKAGE_INSTALLATION.md)
- **Fix conda issues**: [CONDA_ACTIVATION_FIX.md](./CONDA_ACTIVATION_FIX.md)
- **Fix kernel issues**: [JUPYTER_KERNEL_NOT_FOUND.md](./JUPYTER_KERNEL_NOT_FOUND.md)
- **Test GPU**: [TEST_GPU_PYTORCH.md](./TEST_GPU_PYTORCH.md)
- **Make plots**: [MATPLOTLIB_JUPYTER_PLOTTING.md](./MATPLOTLIB_JUPYTER_PLOTTING.md)

---

## 📊 Documentation Structure

```
GPU Jupyter Setup
├── Core Setup & Workflow
│   └── GPU_NOTEBOOK_SETUP.md (441 lines)
│       ├── 6-step setup guide
│       ├── Complete workflow for 3 terminals
│       ├── VS Code connection
│       └── Comprehensive troubleshooting
│
├── Troubleshooting Specialists
│   ├── JUPYTER_KERNEL_NOT_FOUND.md (158 lines)
│   ├── CONDA_ACTIVATION_FIX.md (91 lines)
│   └── CONDA_PACKAGE_INSTALLATION.md (159 lines)
│
├── Quick Reference
│   └── SETUP_README.md (115 lines)
│       └── Quick lookup and orientation
│
├── How-To Guides
│   ├── TEST_GPU_PYTORCH.md (106 lines)
│   └── MATPLOTLIB_JUPYTER_PLOTTING.md (172 lines)
│
└── Index
    └── DOCUMENTATION_INDEX.md (this file)

Total: ~1,350 lines of documentation
```

---

## 🎯 Key Concepts

### SLURM Job Allocation
GPU resources are allocated via SLURM on the HPC cluster. Use:
```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
```

### Module Loading
Required modules must be loaded on each session:
```bash
module load Python/3.12.7 cuda/12.8 Anaconda/2025.06
```

### Conda Environment
Create and activate the `ai-workshop` environment:
```bash
conda create -n ai-workshop python=3.12.7
conda activate ai-workshop
```

### Jupyter Kernel Registration
Jupyter needs to know about the conda environment (requires ipykernel):
```bash
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"
```

### Port Forwarding
Three-step tunnel from local machine → login node → compute node:
1. **Compute node** → Jupyter on port 8888
2. **Login node** → ncat tunnel to port 8889
3. **Local machine** → SSH tunnel to port 8889

### VS Code Connection
Connect to remote Jupyter kernel using full token URL:
```
http://127.0.0.1:8889/?token=YOUR_TOKEN_HERE
```

---

## ✅ Verification Checklist

- [ ] SLURM job allocated with GPU (`srun --gres=gpu:1`)
- [ ] Modules loaded (Python, CUDA, Anaconda)
- [ ] Conda initialized (`conda init bash` + `source`)
- [ ] Environment activated (`conda activate ai-workshop`)
- [ ] PyTorch installed with CUDA support
- [ ] ipykernel installed and kernel registered
- [ ] Jupyter server started
- [ ] Port forwarding active (Terminal 2)
- [ ] SSH tunnel active (Terminal 3)
- [ ] VS Code connected to Jupyter with correct token
- [ ] GPU detected by PyTorch (`torch.cuda.is_available()`)
- [ ] Can run notebook cells and see output

---

## 🔗 Links Between Documents

- GPU_NOTEBOOK_SETUP → JUPYTER_KERNEL_NOT_FOUND (for kernel issues)
- GPU_NOTEBOOK_SETUP → CONDA_ACTIVATION_FIX (for conda issues)
- GPU_NOTEBOOK_SETUP → CONDA_PACKAGE_INSTALLATION (for package install)
- TEST_GPU_PYTORCH → CNN_colab.ipynb (actual test notebook)
- MATPLOTLIB_JUPYTER_PLOTTING → CNN notebook usage examples
- SETUP_README → All specific guides (breadcrumb navigation)

---

## 💡 Tips

1. **Keep terminal windows organized**: Use Terminal 1 (Jupyter), Terminal 2 (Port Forward), Terminal 3 (SSH)
2. **Save your job ID**: Note the SLURM job ID from Terminal 1 prompt
3. **Reuse VS Code connection**: Once connected, you can reconnect by selecting existing server
4. **Check output carefully**: GPU issues often have helpful error messages in terminal output
5. **Use GitHub Copilot**: Integrated in VS Code with your remote GPU kernel for AI-assisted coding
6. **Test incrementally**: Run cells one at a time to debug issues quickly

---

**Last Updated**: November 8, 2026
**Total Lines**: ~1,350 lines of documentation
**Coverage**: Setup, troubleshooting, reference, how-to guides
