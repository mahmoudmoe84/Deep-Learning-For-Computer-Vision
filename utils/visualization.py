"""
Visualization utilities for displaying images and results
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from typing import List, Optional, Tuple


def imshow(img: torch.Tensor, title: Optional[str] = None, denormalize: bool = True):
    """
    Display a single image tensor
    
    Args:
        img: Image tensor (C, H, W)
        title: Optional title for the plot
        denormalize: Whether to denormalize the image
    """
    if denormalize:
        # Denormalize for CIFAR-10
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean
    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def show_batch(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    nrow: int = 8,
    denormalize: bool = True
):
    """
    Display a batch of images in a grid
    
    Args:
        images: Batch of image tensors (B, C, H, W)
        labels: Optional batch of labels
        class_names: Optional list of class names
        nrow: Number of images per row
        denormalize: Whether to denormalize the images
    """
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    if denormalize:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        grid = grid * std + mean
    
    npimg = grid.numpy()
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    if labels is not None and class_names is not None:
        title = ' | '.join([class_names[label] for label in labels[:nrow]])
        plt.title(title)
    
    plt.axis('off')
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training history with loss and accuracy curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: List[str],
    num_images: int = 10
):
    """
    Visualize model predictions with ground truth labels
    
    Args:
        images: Batch of image tensors (B, C, H, W)
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_images: Number of images to display
    """
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = images[i].cpu()
        # Denormalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean
        
        npimg = img.numpy().transpose(1, 2, 0)
        npimg = np.clip(npimg, 0, 1)
        
        axes[i].imshow(npimg)
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
