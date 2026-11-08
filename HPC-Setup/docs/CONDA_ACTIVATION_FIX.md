# Fixing "conda init" Issues

If `conda init` runs but nothing appears to happen, try these solutions in order:

## Solution 1: Manually source the conda initialization (Quickest)

After running `conda init`, manually initialize conda in your current shell:

```bash
# For bash
source ~/miniconda3/etc/profile.d/conda.sh
# OR for newer Anaconda
source /opt/Anaconda/etc/profile.d/conda.sh

# Then activate your environment
conda activate ai-workshop
```

## Solution 2: Check if conda was actually initialized

```bash
# See if conda was added to your shell profile
cat ~/.bashrc | grep conda

# If you see conda initialization lines, close and reopen your terminal
# (or restart your shell)
exec bash
```

## Solution 3: Fully restart your bash shell

After running `conda init`, force bash to reload:

```bash
conda init
exec bash
```

This will close and reopen your bash shell, loading the conda initialization.

## Solution 4: Add conda initialization to startup manually

If the above don't work, add it manually to your `.bashrc`:

```bash
# Add this to the end of ~/.bashrc
if [ -f /opt/Anaconda/etc/profile.d/conda.sh ]; then
    . /opt/Anaconda/etc/profile.d/conda.sh
fi
```

Then reload:
```bash
source ~/.bashrc
conda activate ai-workshop
```

## Complete working sequence

Here's the most reliable sequence:

```bash
# Load modules
module load Python/3.12.7
module load cuda/12.8
module load Anaconda/2025.06

# Initialize conda for bash
conda init bash

# Source the conda initialization immediately
source /opt/Anaconda/etc/profile.d/conda.sh

# Now you can activate
conda activate ai-workshop

# Verify it worked
conda info
```

## Verify conda is active

Once activated, you should see `(ai-workshop)` in your prompt. To verify:

```bash
which python
python --version
echo $CONDA_DEFAULT_ENV
```

All should show your ai-workshop environment.
