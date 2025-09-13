# Efficient MNIST Neural Network

A lightweight convolutional neural network that achieves **95%+ accuracy on MNIST in just 1 epoch** while using **fewer than 25,000 parameters**.

## 🎯 Project Goals

- ✅ **Parameter Efficiency**: Model has fewer than 25,000 parameters (achieved: 8,762)
- ✅ **Fast Learning**: Achieves 95%+ test accuracy in exactly 1 epoch (achieved: 95.29%)
- ⚡ **Quick Training**: Complete training in under 20 seconds on CPU

## 📊 Results

| Metric | Requirement | Achieved |
|--------|-------------|----------|
| Parameters | < 25,000 | **8,762** ✅ |
| Test Accuracy | ≥ 95% | **95.29%** ✅ |
| Training Time | 1 epoch | **16.76 seconds** ✅ |

## 🏗️ Model Architecture

The model uses an efficient CNN architecture:

```
Input (28x28x1)
    ↓
Conv2d(1→8) + BatchNorm + ReLU + MaxPool2d(2x2)
    ↓
Conv2d(8→16) + BatchNorm + ReLU + MaxPool2d(2x2)
    ↓
Conv2d(16→32) + BatchNorm + ReLU + MaxPool2d(2x2)
    ↓
Global Average Pooling
    ↓
Linear(32→64) + ReLU + Dropout(0.3)
    ↓
Linear(64→10)
    ↓
Output (10 classes)
```

**Key Design Decisions:**
- **Global Average Pooling**: Dramatically reduces parameters compared to traditional FC layers
- **Batch Normalization**: Enables faster convergence in 1 epoch
- **Data Augmentation**: Random rotation and translation for better generalization
- **Small Filter Counts**: 8, 16, 32 filters keep parameter count low

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip

### Installation

1. **Clone or download the project files**
2. **Navigate to the project directory**
   ```bash
   cd mnist
   ```

3. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Run the training and testing:**
```bash
python mnist_efficient_model.py
```

The script will:
1. Download the MNIST dataset automatically (if not present)
2. Display model parameter count
3. Train for exactly 1 epoch with progress updates
4. Evaluate on test set
5. Save the model if requirements are met

**Expected Output:**
```
Using device: cpu
Model parameter count: 8,762
Parameter constraint (<25,000): ✓ PASS

==================================================
Starting Training for 1 Epoch
==================================================
Train Batch: 0/469 Loss: 2.306239 Accuracy: 10.94%
Train Batch: 100/469 Loss: 1.257132 Accuracy: 35.66%
...

==================================================
RESULTS
==================================================
Parameters: 8,762 (<25,000: ✓)
Training Time: 16.76 seconds
Test Accuracy: 95.29%
Accuracy Goal (≥95%): ✓ ACHIEVED

✓ Model saved as 'efficient_mnist_model.pth' - All requirements met!
```

## 📁 Project Structure

```
mnist/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── mnist_efficient_model.py  # Main model and training script
├── efficient_mnist_model.pth # Saved trained model (created after running)
├── venv/                     # Virtual environment
└── data/                     # MNIST dataset (auto-downloaded)
    ├── MNIST/
    │   └── raw/
    │       ├── train-images-idx3-ubyte
    │       ├── train-labels-idx1-ubyte
    │       ├── t10k-images-idx3-ubyte
    │       └── t10k-labels-idx1-ubyte
```

## 🔧 Dependencies

- **PyTorch** (≥2.0.0): Deep learning framework
- **torchvision** (≥0.15.0): Computer vision utilities and datasets
- **numpy** (≥1.21.0): Numerical computing

## 🧠 Technical Details

### Key Optimizations for 1-Epoch Performance:

1. **Data Augmentation**: Random rotations (±10°) and translations (±10%) increase effective dataset size
2. **Batch Normalization**: Stabilizes training and enables higher learning rates
3. **Adam Optimizer**: Adaptive learning rate with weight decay (1e-4)
4. **Global Average Pooling**: Reduces parameters while maintaining spatial information
5. **Appropriate Learning Rate**: 0.001 balances fast learning with stability

### Parameter Efficiency Techniques:

1. **Small Filter Counts**: Progressive increase (8→16→32) instead of typical powers of 2
2. **Global Average Pooling**: Eliminates large fully connected layers
3. **Compact FC Layers**: Only 32→64→10 instead of typical 512/1024-wide layers
4. **No Bias in Conv Layers**: BatchNorm makes bias redundant

## 🎛️ Customization

### Modify Hyperparameters:
```python
# In mnist_efficient_model.py
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Learning rate
train_loader = DataLoader(..., batch_size=128, ...)  # Batch size
self.dropout = nn.Dropout(0.3)  # Dropout rate
```

### Adjust Data Augmentation:
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotation angle
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
    # Add more augmentations here
])
```

## 📈 Performance Analysis

The model achieves high accuracy quickly due to:

- **Efficient Architecture**: Every layer contributes meaningfully without redundancy
- **Smart Regularization**: Dropout and data augmentation prevent overfitting
- **Optimization**: Adam with proper learning rate and weight decay
- **Normalization**: BatchNorm enables stable training at higher learning rates

## 🚀 Running on GPU

For faster training, the model automatically uses GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Expected GPU training time: ~3-5 seconds

## 🛠️ Troubleshooting

**Common Issues:**

1. **ImportError: No module named 'torch'**
   - Ensure virtual environment is activated
   - Run: `pip install -r requirements.txt`

2. **CUDA out of memory**
   - Reduce batch size in `get_data_loaders(batch_size=64)`

3. **Slow training**
   - Check if GPU is available: `torch.cuda.is_available()`
   - Increase batch size if you have more memory

4. **Lower accuracy than expected**
   - Ensure data augmentation is enabled
   - Check that batch normalization is working
   - Try running multiple times (some variance is normal)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements!

---

**Happy Learning! 🎉**
