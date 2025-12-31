# Data Directory

This directory is used for storing datasets. Datasets will be automatically downloaded when you run the example scripts or use the data loader utilities.

## Structure

```
data/
├── cifar-10-batches-py/    # CIFAR-10 dataset (auto-downloaded)
├── MNIST/                   # MNIST dataset (auto-downloaded)
└── custom/                  # Your custom datasets
```

## Automatic Downloads

When you run examples or use data loaders, datasets will be automatically downloaded:

```python
from utils.data_loader import get_cifar10_loaders

# This will download CIFAR-10 to ./data if not present
train_loader, test_loader = get_cifar10_loaders(data_dir='./data')
```

## Custom Datasets

To use your own image dataset, organize it as follows:

```
data/custom/your_dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...
```

Then load it with:

```python
from utils.data_loader import CustomImageDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

dataset = CustomImageDataset(
    root_dir='./data/custom/your_dataset',
    transform=transform
)
```

## Note

This directory is in `.gitignore` to avoid committing large dataset files. Only this README is tracked by git.
