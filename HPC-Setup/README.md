# HPC GPU Jupyter Notebook Setup

Complete guide and documentation for running GPU-accelerated Jupyter notebooks on HPC clusters with VS Code and GitHub Copilot integration.

## 📁 Folder Structure

```
HPC-Setup/
├── README.md                          # This file
├── verify_gpu_setup.ipynb            # GPU verification notebook
└── docs/                              # All documentation
    ├── DOCUMENTATION_INDEX.md         # Start here for navigation
    ├── SETUP_README.md               # Quick reference and 3-step start
    ├── GPU_NOTEBOOK_SETUP.md         # Complete 6-step setup guide
    ├── JUPYTER_KERNEL_NOT_FOUND.md   # Kernel troubleshooting
    ├── CONDA_ACTIVATION_FIX.md       # Conda activation issues
    ├── CONDA_PACKAGE_INSTALLATION.md # Package installation guide
    ├── MATPLOTLIB_JUPYTER_PLOTTING.md # Plotting in remote notebooks
    └── TEST_GPU_PYTORCH.md           # GPU testing with CNN notebook
```

## 🚀 Quick Start

### First Time Setup

1. **Read**: [docs/SETUP_README.md](./docs/SETUP_README.md) (5 min overview)
2. **Follow**: [docs/GPU_NOTEBOOK_SETUP.md](./docs/GPU_NOTEBOOK_SETUP.md) (Complete workflow)
3. **Verify**: Run `verify_gpu_setup.ipynb` after connecting your kernel

### Already Have a Setup

- **Connection Issues**: [docs/GPU_NOTEBOOK_SETUP.md](./docs/GPU_NOTEBOOK_SETUP.md) (Section: VS Code Connection)
- **Kernel Problems**: [docs/JUPYTER_KERNEL_NOT_FOUND.md](./docs/JUPYTER_KERNEL_NOT_FOUND.md)
- **Conda Issues**: [docs/CONDA_ACTIVATION_FIX.md](./docs/CONDA_ACTIVATION_FIX.md)
- **Installing Packages**: [docs/CONDA_PACKAGE_INSTALLATION.md](./docs/CONDA_PACKAGE_INSTALLATION.md)
- **Making Plots**: [docs/MATPLOTLIB_JUPYTER_PLOTTING.md](./docs/MATPLOTLIB_JUPYTER_PLOTTING.md)

## 📖 Documentation Files

### Main Documentation

#### [docs/SETUP_README.md](./docs/SETUP_README.md)
**Quick reference for beginners** (115 lines)
- 3-step quick start overview
- Common issues table
- Key commands reference
- File navigation guide

**When to use**: Need a quick overview before diving into details

---

#### [docs/GPU_NOTEBOOK_SETUP.md](./docs/GPU_NOTEBOOK_SETUP.md)
**Complete end-to-end setup guide** (441 lines)
- 6-step setup procedure
- Module loading and CUDA configuration
- Conda environment creation with PyTorch
- Jupyter server setup
- Port forwarding configuration
- VS Code kernel connection (primary method)
- GPU verification code
- Comprehensive troubleshooting section

**When to use**: Initial setup, need complete workflow, reference for any step

---

### Troubleshooting Guides

#### [docs/JUPYTER_KERNEL_NOT_FOUND.md](./docs/JUPYTER_KERNEL_NOT_FOUND.md)
**Fix missing conda kernel in VS Code** (158 lines)
- Why conda kernel doesn't auto-register
- 5-step ipykernel installation
- Kernel registration and verification
- Multiple solution approaches

**When to use**: "Kernel not found" error in VS Code

---

#### [docs/CONDA_ACTIVATION_FIX.md](./docs/CONDA_ACTIVATION_FIX.md)
**Fix conda activation issues** (91 lines)
- Why `conda init` appears to do nothing
- 4 different solution approaches
- Manual shell initialization
- Working complete sequence

**When to use**: Can't activate conda environment in terminal

---

#### [docs/CONDA_PACKAGE_INSTALLATION.md](./docs/CONDA_PACKAGE_INSTALLATION.md)
**Install and manage Python packages** (159 lines)
- Quick package installation with conda
- Common data science packages
- Installing PyTorch with CUDA support
- Batch package installation
- Troubleshooting conflicts

**When to use**: Need to install packages (scikit-learn, pytorch, etc.)

---

### How-To Guides

#### [docs/TEST_GPU_PYTORCH.md](./docs/TEST_GPU_PYTORCH.md)
**Test GPU functionality with real models** (106 lines)
- GPU verification code samples
- CNN notebook location in AI-Workshop-UNTRM
- Expected output (GPU vs CPU performance)
- Performance benchmarks
- Links to other GPU notebooks

**When to use**: Want to test GPU is actually working with real computations

---

#### [docs/MATPLOTLIB_JUPYTER_PLOTTING.md](./docs/MATPLOTLIB_JUPYTER_PLOTTING.md)
**Use matplotlib in remote Jupyter notebooks** (172 lines)
- `%matplotlib inline` magic command
- Common plotting issues and solutions
- Best practices for remote display
- CNN notebook plotting examples
- Troubleshooting remote plot rendering

**When to use**: Plots not displaying or need plotting guidance

---

### Navigation

#### [docs/DOCUMENTATION_INDEX.md](./docs/DOCUMENTATION_INDEX.md)
**Complete documentation map** (300+ lines)
- Overview of all files
- Quick start paths for different scenarios
- Documentation structure
- Key concepts reference
- Verification checklist
- Cross-reference links

**When to use**: Need to navigate all documentation or find specific topics

---

## 🧪 Verification Notebook

### [verify_gpu_setup.ipynb](./verify_gpu_setup.ipynb)

A self-contained Jupyter notebook that verifies your GPU setup is working correctly.

**What it checks:**
1. System information (hostname, Python version, SLURM job ID)
2. GPU hardware (via `nvidia-smi`)
3. PyTorch installation and CUDA availability
4. GPU memory access and allocation
5. GPU computation speed vs CPU
6. Device transfer (CPU ↔ GPU)
7. Overall setup status with summary

**How to use:**
```bash
# After connecting your remote GPU kernel in VS Code:
# 1. Open verify_gpu_setup.ipynb
# 2. Select "AI Workshop (GPU)" kernel
# 3. Run all cells (Ctrl+Shift+Enter or Run menu)
# 4. Check the Summary section for status
```

**Expected output when everything works:**
```
✅ ALL CHECKS PASSED - GPU SETUP IS WORKING!
You can now run GPU-accelerated notebooks.
```

---

## 🎯 Key Concepts

### SLURM GPU Allocation
Request GPU resources from the HPC cluster:
```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
```

### Module Loading Stack
Load required modules in sequence:
```bash
module load Python/3.12.7 cuda/12.8 Anaconda/2025.06
```

### Conda Environment with GPU Support
Create and activate conda environment with PyTorch:
```bash
conda create -n ai-workshop python=3.12.7
conda activate ai-workshop
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Jupyter Kernel Registration
Register conda environment as Jupyter kernel:
```bash
python -m ipykernel install --user --name ai-workshop --display-name "AI Workshop (GPU)"
```

### Port Forwarding (3 Steps)
1. **Compute node**: Jupyter server on 8888
2. **Login node**: Forward port 8889
3. **Local machine**: SSH tunnel to port 8889

### VS Code Connection
Access remote Jupyter in VS Code using token:
```
http://127.0.0.1:8889/?token=YOUR_TOKEN_HERE
```

---

## ✅ Verification Checklist

Before running your first GPU notebook:

- [ ] SLURM job allocated with GPU
- [ ] Modules loaded (Python, CUDA, Anaconda)
- [ ] Conda environment created
- [ ] Conda environment activated
- [ ] PyTorch installed with CUDA support
- [ ] ipykernel installed and registered
- [ ] Jupyter server running
- [ ] Port forwarding active
- [ ] SSH tunnel active
- [ ] VS Code connected to Jupyter
- [ ] GPU recognized by PyTorch
- [ ] `verify_gpu_setup.ipynb` passes all checks

---

## 🔧 Common Issues Quick-Fix

| Issue | Solution |
|-------|----------|
| Kernel not appearing in VS Code | See [JUPYTER_KERNEL_NOT_FOUND.md](./docs/JUPYTER_KERNEL_NOT_FOUND.md) |
| Can't activate conda | See [CONDA_ACTIVATION_FIX.md](./docs/CONDA_ACTIVATION_FIX.md) |
| Need to install packages | See [CONDA_PACKAGE_INSTALLATION.md](./docs/CONDA_PACKAGE_INSTALLATION.md) |
| Plots not displaying | See [MATPLOTLIB_JUPYTER_PLOTTING.md](./docs/MATPLOTLIB_JUPYTER_PLOTTING.md) |
| Want to test GPU | Run [verify_gpu_setup.ipynb](./verify_gpu_setup.ipynb) or see [TEST_GPU_PYTORCH.md](./docs/TEST_GPU_PYTORCH.md) |
| General setup help | See [GPU_NOTEBOOK_SETUP.md](./docs/GPU_NOTEBOOK_SETUP.md) |

---

## 📚 Learning Path

### Beginner (First-time setup)
1. Read: [SETUP_README.md](./docs/SETUP_README.md)
2. Follow: [GPU_NOTEBOOK_SETUP.md](./docs/GPU_NOTEBOOK_SETUP.md) complete workflow
3. Verify: Run [verify_gpu_setup.ipynb](./verify_gpu_setup.ipynb)

### Intermediate (Troubleshooting)
1. Use Common Issues table (above) to find your problem
2. Go to specific troubleshooting guide
3. Follow step-by-step solution
4. Verify with [verify_gpu_setup.ipynb](./verify_gpu_setup.ipynb)

### Advanced (Deep understanding)
1. Read [DOCUMENTATION_INDEX.md](./docs/DOCUMENTATION_INDEX.md) for overview
2. Study [GPU_NOTEBOOK_SETUP.md](./docs/GPU_NOTEBOOK_SETUP.md) complete guide
3. Explore all docs/ folder guides
4. Review troubleshooting sections for edge cases

---

## 💡 Tips & Best Practices

1. **Keep terminals organized**: Use 3 terminals for Jupyter, port forwarding, and SSH tunnel
2. **Save your job ID**: Note the SLURM job ID from the prompt
3. **Test incrementally**: Run notebook cells one at a time to debug issues
4. **Use GitHub Copilot**: Your VS Code connection enables AI-assisted coding with Copilot
5. **Monitor GPU**: Watch `nvidia-smi` in terminal for GPU utilization during runs
6. **Clean up**: Exit Jupyter and close tunnels when finished to free resources

---

## 📞 Troubleshooting Flow

```
Problem?
  │
  ├─→ Kernel not found? → JUPYTER_KERNEL_NOT_FOUND.md
  ├─→ Conda issues? → CONDA_ACTIVATION_FIX.md
  ├─→ Need packages? → CONDA_PACKAGE_INSTALLATION.md
  ├─→ Plots not showing? → MATPLOTLIB_JUPYTER_PLOTTING.md
  ├─→ Want to test GPU? → verify_gpu_setup.ipynb or TEST_GPU_PYTORCH.md
  └─→ General setup? → GPU_NOTEBOOK_SETUP.md
```

---

## 📋 File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| SETUP_README.md | 115 | Quick reference |
| GPU_NOTEBOOK_SETUP.md | 441 | Complete guide |
| JUPYTER_KERNEL_NOT_FOUND.md | 158 | Kernel troubleshooting |
| CONDA_ACTIVATION_FIX.md | 91 | Conda issues |
| CONDA_PACKAGE_INSTALLATION.md | 159 | Package management |
| MATPLOTLIB_JUPYTER_PLOTTING.md | 172 | Plotting guide |
| TEST_GPU_PYTORCH.md | 106 | GPU testing |
| DOCUMENTATION_INDEX.md | 300+ | Navigation index |
| verify_gpu_setup.ipynb | 7 cells | Verification notebook |
| **TOTAL** | **~1,550** | **Complete documentation** |

---

## 🔄 For Future Repository Integration

This entire `HPC-Setup/` folder is designed to be:
- ✅ Copied into a GitHub repository
- ✅ Shared with team members
- ✅ Updated as cluster configurations change
- ✅ Customized for different HPC systems
- ✅ Integrated with project-specific notebooks

### To integrate into a repository:
```bash
# Copy entire folder into repo
cp -r HPC-Setup/ your-repo/docs/
# or
cp -r HPC-Setup/ your-repo/setup/
```

Then update repository README to reference:
```markdown
## GPU Notebook Setup

See [HPC-Setup/README.md](./HPC-Setup/README.md) for complete setup and troubleshooting guide.
```

---

## ℹ️ About This Setup

- **Target System**: HPC cluster with NVIDIA GPUs
- **Environment**: SLURM job scheduler, module system (Python/CUDA/Anaconda)
- **IDE**: VS Code with remote development
- **Integration**: GitHub Copilot in VS Code chat
- **Language**: Python 3.12.7 with PyTorch
- **Documentation**: Markdown + Jupyter notebooks

---

## 📝 Version History

- **v1.0** - Initial comprehensive documentation
  - 8 markdown documentation files
  - 1 GPU verification notebook
  - Complete setup workflows
  - Comprehensive troubleshooting
  - VS Code + GitHub Copilot integration

---

**Last Updated**: November 8, 2026  
**Status**: Complete and ready for use  
**Ready for**: Repository integration and team sharing
