# Examples

This directory contains complete example implementations for various computer vision tasks.

## Available Examples

### 1. Image Classification

#### CIFAR-10 Classification (`cifar10_classification.py`)
Train a CNN model on the CIFAR-10 dataset containing 10 classes of images.

**Usage:**
```bash
cd examples
python cifar10_classification.py
```

**Features:**
- SimpleCNN architecture
- Data augmentation
- Learning rate scheduling
- Model checkpointing
- Training visualization

#### MNIST Classification (`mnist_classification.py`)
Train a simple CNN on the MNIST handwritten digit dataset.

**Usage:**
```bash
cd examples
python mnist_classification.py
```

**Features:**
- Lightweight CNN architecture
- Fast training (10 epochs)
- >99% accuracy achievable

## Running Examples

1. Make sure you have installed all dependencies:
```bash
pip install -r ../requirements.txt
```

2. Navigate to the examples directory:
```bash
cd examples
```

3. Run any example script:
```bash
python cifar10_classification.py
```

## Expected Output

Each example will:
- Download the dataset automatically (if not present)
- Display training progress with loss and accuracy
- Save the best model checkpoint
- Generate training history plots
- Print final results

## Customization

You can modify hyperparameters at the top of each script:
- `BATCH_SIZE`: Number of samples per batch
- `LEARNING_RATE`: Learning rate for optimizer
- `NUM_EPOCHS`: Number of training epochs
- `NUM_CLASSES`: Number of output classes

## Tips

- Use a GPU for faster training (CUDA-enabled PyTorch)
- Increase `NUM_EPOCHS` for better results
- Experiment with different model architectures from `models/`
- Try different optimizers and learning rate schedules
