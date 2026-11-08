# Using Matplotlib in Jupyter Notebooks with GPU

## Quick Setup

If you're seeing plots not displaying in your Jupyter notebook, add this at the beginning of your notebook:

```python
%matplotlib inline
import matplotlib.pyplot as plt
```

## For Remote Jupyter Servers (Your GPU Setup)

When using Jupyter on a remote GPU server accessed via VS Code, use:

```python
%matplotlib inline
import matplotlib.pyplot as plt

# For better quality plots
plt.rcParams.update({"font.size": 14, "lines.linewidth": 3})
```

## Common Issues and Solutions

### Issue: Plots Not Displaying

**Solution**: Add the magic command at the start of your notebook:

```python
%matplotlib inline
import matplotlib.pyplot as plt
```

The `%matplotlib inline` magic command tells Jupyter to display plots directly in the notebook.

### Issue: Plots Display but Are Small/Unclear

**Solution**: Adjust figure size and font:

```python
import matplotlib.pyplot as plt

# Larger figures
plt.figure(figsize=(12, 6))

# Or adjust defaults for all plots
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 14
```

### Issue: Multiple Plots on Same Figure

**Solution**: Use subplots:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x1, y1)
axes[0].set_title("Plot 1")
axes[1].plot(x2, y2)
axes[1].set_title("Plot 2")
plt.show()
```

## CNN Notebook Plotting

The CNN notebook uses matplotlib for:

```python
import matplotlib.pyplot as plt

# Display bean leaf images
plt.imshow(image_array)
plt.title("Bean Leaf")
plt.show()

# Plot training history
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix, display_labels=class_names).plot()
plt.show()
```

## Best Practices for Remote GPU Notebooks

### 1. Always Start with Magic Command

```python
%matplotlib inline
import matplotlib.pyplot as plt
```

### 2. Configure for Remote Display

```python
# At top of notebook
%matplotlib inline
import matplotlib.pyplot as plt

# Set good defaults
plt.rcParams.update({
    "font.size": 14,
    "lines.linewidth": 3,
    "figure.figsize": (12, 6)
})
```

### 3. Use plt.show() Explicitly

```python
# Good practice
plt.plot(data)
plt.show()  # Ensures plot displays

# Or use semicolon to suppress output
plt.plot(data);
```

### 4. Clear Plots Between Cells

```python
# Between plots
plt.clf()  # Clear current figure
# or
plt.close('all')  # Close all figures
```

## Complete Example for CNN Notebook

```python
# Cell 1: Setup
%matplotlib inline
import matplotlib.pyplot as plt
import torch
import numpy as np

plt.rcParams.update({"font.size": 14, "lines.linewidth": 3})

# Cell 2: Load and display images
from PIL import Image

img_array = np.random.rand(100, 100, 3)  # Example image
plt.figure(figsize=(6, 6))
plt.imshow(img_array)
plt.title("Sample Image")
plt.axis('off')
plt.show()

# Cell 3: Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'])
plt.plot(history['val_acc'])
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()
```

## Troubleshooting Remote Plotting

If plots still don't appear:

1. **Check matplotlib is installed**:
   ```python
   import matplotlib
   print(matplotlib.__version__)
   ```

2. **Verify backend**:
   ```python
   import matplotlib
   print(matplotlib.get_backend())
   # Should output 'module://ipykernel.pylab.backend_inline'
   ```

3. **Reinstall matplotlib**:
   ```bash
   pip install --upgrade matplotlib
   ```

4. **Check VS Code has correct kernel selected**:
   - Should show "AI Workshop (GPU)" in kernel indicator
   - Check it's running on correct compute node (e.g., g001)

## Performance Note

Plotting on a remote GPU node is generally fast because:
- The computation (generating the plot data) happens on GPU
- Only the final image is transmitted back to your local VS Code

So don't hesitate to create multiple plots for analysis!
