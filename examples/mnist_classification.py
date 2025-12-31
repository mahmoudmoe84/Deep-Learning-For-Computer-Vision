"""
MNIST Digit Classification Example

This example demonstrates how to train a simple CNN on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import get_mnist_loaders
from utils.training import train_one_epoch, evaluate, save_checkpoint


class MNISTNet(nn.Module):
    """
    Simple CNN for MNIST digit classification
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def main():
    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading MNIST dataset...')
    train_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
        data_dir='../data'
    )
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Initialize model
    model = MNISTNet().to(device)
    print(f'\nModel: {model.__class__.__name__}')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    # Training loop
    print('\nStarting training...')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{NUM_EPOCHS}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # Print results
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                'mnist_best_model.pth'
            )
            print(f'New best model saved! Accuracy: {best_acc:.2f}%')
    
    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
