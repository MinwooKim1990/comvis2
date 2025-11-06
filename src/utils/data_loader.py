"""
Data loading utilities for Active Visual Search.

Handles CIFAR-10 dataset loading and preprocessing.
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def load_cifar10(
    data_dir: str = './data',
    train: bool = True,
    download: bool = True
) -> datasets.CIFAR10:
    """
    Load CIFAR-10 dataset.

    Args:
        data_dir: Directory to store/load data
        train: If True, load training set, else test set
        download: If True, download dataset if not present

    Returns:
        CIFAR-10 dataset
    """
    # Basic transform (just convert to tensor)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )

    return dataset


def create_filtered_dataset(
    dataset: datasets.CIFAR10,
    target_classes: list,
    max_samples: Optional[int] = None
) -> Subset:
    """
    Create subset of dataset containing only target classes.

    Args:
        dataset: Full CIFAR-10 dataset
        target_classes: List of class indices to include
        max_samples: Maximum number of samples (None = all)

    Returns:
        Filtered dataset subset
    """
    # Find indices of target classes
    indices = []
    class_counts = {c: 0 for c in target_classes}

    for idx, (_, label) in enumerate(dataset):
        if label in target_classes:
            indices.append(idx)
            class_counts[label] += 1

            if max_samples and len(indices) >= max_samples:
                break

    print(f"Filtered dataset: {len(indices)} samples")
    print(f"Class distribution: {class_counts}")

    return Subset(dataset, indices)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training data
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Shuffle with seed
    np.random.seed(seed)
    np.random.shuffle(indices)

    # Split
    split_idx = int(train_ratio * dataset_size)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")

    return train_dataset, val_dataset


def get_class_names() -> list:
    """Get CIFAR-10 class names."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def prepare_datasets(config: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare train, val, and test datasets based on config.

    Args:
        config: Configuration dictionary

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    data_dir = config['dataset'].get('data_dir', './data')
    target_classes = config['dataset']['target_classes']
    train_size = config['dataset'].get('train_size', None)
    train_ratio = config['dataset'].get('train_split', 0.8)

    print("\n=== Preparing Datasets ===")
    print(f"Data directory: {data_dir}")
    print(f"Target classes: {[get_class_names()[c] for c in target_classes]}")

    # Load train and test sets
    print("\nLoading CIFAR-10 training set...")
    train_full = load_cifar10(data_dir=data_dir, train=True, download=True)

    print("\nLoading CIFAR-10 test set...")
    test_full = load_cifar10(data_dir=data_dir, train=False, download=True)

    # Filter by target classes
    print("\nFiltering training set...")
    train_filtered = create_filtered_dataset(
        train_full, target_classes, max_samples=train_size
    )

    print("\nFiltering test set...")
    test_dataset = create_filtered_dataset(
        test_full, target_classes, max_samples=None
    )

    # Split train into train/val
    print("\nSplitting train/val...")
    train_dataset, val_dataset = split_dataset(
        train_filtered, train_ratio=train_ratio, seed=config['experiment']['seed']
    )

    print("\n=== Dataset Preparation Complete ===\n")

    return train_dataset, val_dataset, test_dataset


def test_data_loader():
    """Test data loading functionality."""
    print("\n=== Testing Data Loader ===\n")

    # Test basic loading
    dataset = load_cifar10(data_dir='./data', train=True, download=True)
    print(f"Loaded CIFAR-10: {len(dataset)} samples")

    # Test filtering
    target_classes = [0, 1, 2, 3, 4]  # airplane, car, bird, cat, deer
    filtered = create_filtered_dataset(dataset, target_classes, max_samples=1000)
    print(f"Filtered dataset: {len(filtered)} samples")

    # Test split
    train_set, val_set = split_dataset(filtered, train_ratio=0.8)

    # Sample one image
    img, label = train_set[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label} ({get_class_names()[label]})")

    print("\n=== Test Complete ===\n")


if __name__ == '__main__':
    test_data_loader()
