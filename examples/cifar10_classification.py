"""
CIFAR-10 Image Classification Example

This example demonstrates how to train a CNN model on the CIFAR-10 dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import SimpleCNN
from utils.data_loader import get_cifar10_loaders
from utils.training import train_one_epoch, evaluate, save_checkpoint
from utils.visualization import plot_training_history


def main():
    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    NUM_CLASSES = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
        data_dir='../data'
    )
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Initialize model
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    print(f'\nModel: {model.__class__.__name__}')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
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
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print results
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                'best_model.pth'
            )
            print(f'New best model saved! Accuracy: {best_acc:.2f}%')
    
    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'{"="*60}')
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path='training_history.png'
    )


if __name__ == '__main__':
    main()
