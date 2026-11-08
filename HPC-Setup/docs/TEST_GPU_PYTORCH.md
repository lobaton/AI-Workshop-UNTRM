# Testing GPU with PyTorch - CNN Notebook

## Quick Test

The **CNN_colab.ipynb** notebook in your AI-Workshop-UNTRM folder is perfect for testing GPU functionality with PyTorch.

### Location
```
/home/indes/investigadores/edgar.lobaton/AI-Workshop-UNTRM/session2_inside_the_black_box/src/CNN_colab.ipynb
```

### What It Tests

The notebook:
1. **Checks GPU availability**: `device = "cuda" if torch.cuda.is_available() else "cpu"`
2. **Prints device used**: Shows "GPU" or "CPU (training will be slower)"
3. **Trains a CNN model**: Performs actual GPU computation on bean leaf disease classification
4. **Measures training time**: Reports how long training took (GPU should be noticeably faster)

### How to Run

#### Option 1: Using Your Setup (Recommended)

1. **Start your GPU allocation and Jupyter server** (use your documented setup)
2. **In VS Code**, connect to the remote kernel using your setup documentation
3. **Open the notebook**:
   ```
   /home/indes/investigadores/edgar.lobaton/AI-Workshop-UNTRM/session2_inside_the_black_box/src/CNN_colab.ipynb
   ```
4. **Run the cells in order**:
   - Cell 2: Setup and GPU check (should print "Using GPU")
   - Cell 4: Configure settings (defaults are fine)
   - Cell 5: Train the model (watch the training time)

#### Option 2: Quick Local Test (CPU Only)

If you want a quick GPU availability test without running the full notebook:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
    
    # Quick GPU test
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("GPU test passed ✓")
else:
    print("GPU not available")
```

### Expected Output (GPU)

```
Using GPU
train      : 1034 images
validation : 124 images
test       : 128 images

The network has 78,598 parameters
Training took 45 seconds
```

### Expected Output (CPU)

```
Using CPU (training will be slower)
train      : 1034 images
validation : 124 images
test       : 128 images

The network has 78,598 parameters
Training took 300+ seconds
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Using CPU (training will be slower)" | GPU not detected - check CUDA module loaded |
| Import error: `cnn_common` | The notebook downloads it automatically - check internet connectivity or network access |
| Training is very slow | GPU isn't being used - check CUDA setup |
| Out of memory | GPU memory full - reduce N_TRAIN or N_FILTERS |

### Performance Expectations

- **GPU (Tesla V100+)**: ~30-60 seconds per epoch
- **GPU (other models)**: ~30-120 seconds per epoch  
- **CPU**: Several minutes per epoch

If your GPU training is taking longer than a few seconds per epoch, something may be wrong.

### Other Notebooks Available

The AI-Workshop-UNTRM also includes:
- `Attention_colab.ipynb` - Attention mechanisms
- `Autoencoder_colab.ipynb` - Autoencoders
- `RNN_colab.ipynb` - Recurrent neural networks
- `DNA_motif_colab.ipynb` - DNA sequence analysis

All support GPU acceleration.
