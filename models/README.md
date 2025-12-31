# Models

This directory contains neural network model architectures for computer vision tasks.

## Available Models

### Convolutional Neural Networks (`cnn.py`)

#### 1. SimpleCNN
A straightforward CNN architecture suitable for datasets like CIFAR-10.

**Architecture:**
- 3 Convolutional blocks with BatchNorm and MaxPooling
- 2 Fully connected layers
- Dropout for regularization

**Usage:**
```python
from models.cnn import SimpleCNN

model = SimpleCNN(num_classes=10)
```

**Parameters:**
- Total parameters: ~300K
- Input size: (B, 3, 32, 32)
- Output size: (B, num_classes)

#### 2. SimpleResNet
A residual network architecture with skip connections for better gradient flow.

**Architecture:**
- Initial convolution layer
- 3 residual layers with 2 blocks each
- Adaptive average pooling
- Fully connected output layer

**Usage:**
```python
from models.cnn import SimpleResNet

model = SimpleResNet(num_classes=10)
```

**Parameters:**
- Total parameters: ~600K
- Better performance than SimpleCNN
- More stable training

#### 3. ResidualBlock
Building block for ResNet architectures with skip connections.

**Features:**
- Batch normalization
- Skip connections (identity mapping)
- Handles dimension changes with 1x1 convolutions

## Using Models

### Basic Usage

```python
import torch
from models.cnn import SimpleCNN, SimpleResNet

# Create model
model = SimpleCNN(num_classes=10)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Forward pass
x = torch.randn(32, 3, 32, 32).to(device)  # Batch of 32 images
output = model(x)  # Shape: (32, 10)
```

### Training

```python
import torch.nn as nn
import torch.optim as optim

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## Custom Models

You can create your own models by:

1. Inheriting from `nn.Module`
2. Defining layers in `__init__`
3. Implementing the `forward` method

Example:
```python
import torch.nn as nn

class MyCustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCustomCNN, self).__init__()
        # Define your layers here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Define forward pass
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## Model Selection Guide

- **SimpleCNN**: Good starting point, fast training, moderate accuracy
- **SimpleResNet**: Better accuracy, slightly slower, more stable training
- **Custom Models**: For specific requirements or research

## Tips

- Start with SimpleCNN for quick experiments
- Use SimpleResNet for better performance
- Add BatchNorm for training stability
- Use Dropout to prevent overfitting
- Consider pretrained models (ResNet, VGG) for transfer learning
