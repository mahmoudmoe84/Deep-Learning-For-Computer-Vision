# Getting Started Guide

This guide will help you get started with the Deep Learning for Computer Vision repository.

## Prerequisites

- Python 3.8 or higher
- Basic understanding of Python and deep learning concepts
- (Optional) CUDA-capable GPU for faster training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mahmoudmoe84/Deep-Learning-For-Computer-Vision.git
cd Deep-Learning-For-Computer-Vision
```

### 2. Create Virtual Environment (Recommended)

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (with CUDA support if available)
- torchvision
- NumPy, Matplotlib, Pillow
- OpenCV, scikit-learn
- Jupyter Notebook
- And other dependencies

### 4. Verify Installation

```python
import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"torchvision version: {torchvision.__version__}")
```

## Quick Start

### Option 1: Run an Example Script

The fastest way to get started is to run one of the example scripts:

```bash
cd examples
python mnist_classification.py
```

This will:
1. Download the MNIST dataset
2. Train a CNN model
3. Display training progress
4. Save the best model

### Option 2: Use Jupyter Notebook

For interactive learning, use the provided notebooks:

```bash
jupyter notebook
```

Then open `notebooks/image_classification_tutorial.ipynb`

### Option 3: Write Your Own Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import SimpleCNN
from utils.data_loader import get_cifar10_loaders
from utils.training import train_one_epoch, evaluate

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar10_loaders(batch_size=32)

# Model
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch}: Val Acc = {val_acc:.2f}%')
```

## Project Structure

```
Deep-Learning-For-Computer-Vision/
│
├── examples/              # Complete training examples
│   ├── cifar10_classification.py
│   └── mnist_classification.py
│
├── models/               # Neural network architectures
│   ├── cnn.py           # CNN models (SimpleCNN, SimpleResNet)
│   └── __init__.py
│
├── utils/               # Utility functions
│   ├── data_loader.py   # Data loading utilities
│   ├── visualization.py # Plotting and visualization
│   ├── training.py      # Training and evaluation
│   └── __init__.py
│
├── notebooks/           # Jupyter notebooks
│   └── image_classification_tutorial.ipynb
│
├── data/               # Dataset directory (auto-created)
├── docs/               # Documentation
├── requirements.txt    # Python dependencies
└── README.md          # Main documentation
```

## Next Steps

### 1. Learn the Basics
- Start with the MNIST example (`examples/mnist_classification.py`)
- Work through the Jupyter notebook tutorial
- Understand the training loop and model architecture

### 2. Experiment
- Modify hyperparameters (learning rate, batch size)
- Try different model architectures
- Add data augmentation techniques

### 3. Build Projects
- Use CIFAR-10 for more challenging classification
- Create custom datasets with `CustomImageDataset`
- Implement your own model architectures

### 4. Advanced Topics
- Transfer learning with pretrained models
- Multi-GPU training
- Model optimization and quantization

## Common Issues

### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

### Slow Training
- Use GPU if available
- Increase `num_workers` in data loaders
- Use smaller model for testing

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version`

## Learning Resources

### Tutorials
- PyTorch Official Tutorials: https://pytorch.org/tutorials/
- Deep Learning Book: https://www.deeplearningbook.org/
- CS231n Stanford: http://cs231n.stanford.edu/

### Datasets
- MNIST: Handwritten digits (60K training, 10K test)
- CIFAR-10: 10 classes of images (50K training, 10K test)
- ImageNet: Large-scale image dataset

## Getting Help

- Check the documentation in each directory
- Review example scripts for usage patterns
- Open an issue on GitHub for bugs or questions
- Read error messages carefully

## Tips for Success

1. **Start Small**: Begin with MNIST, then move to CIFAR-10
2. **Understand Basics**: Know how models, optimizers, and data loaders work
3. **Experiment**: Try different architectures and hyperparameters
4. **Monitor Training**: Use TensorBoard or plots to track progress
5. **Save Checkpoints**: Always save your best models
6. **Read Code**: Study the utility functions and example scripts

## Contributing

Contributions are welcome! Feel free to:
- Add new model architectures
- Create additional examples
- Improve documentation
- Fix bugs or issues

Happy Learning! 🚀
