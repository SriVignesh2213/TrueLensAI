"""
TrueLens AI — Dataset Preparation Pipeline

Handles dataset loading, preprocessing, augmentation, and splitting
for training the AI image detection models.

Supports:
    - Binary classification (Real vs AI-Generated)
    - Multi-class classification (Real, GAN, Diffusion, Manipulated)
    - Custom dataset directory structures
    - Automated train/val/test splitting
    - Data augmentation for training robustness

Expected Directory Structure:
    dataset/
    ├── real/
    │   ├── img_001.jpg
    │   └── ...
    └── ai_generated/
        ├── img_001.jpg
        └── ...

    OR for multi-class:
    dataset/
    ├── real/
    ├── gan/
    ├── diffusion/
    └── manipulated/

Author: TrueLens AI Team
License: MIT
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training data augmentation transforms.

    Augmentation strategy designed for forensic detection robustness:
        - Random resized crop with varying scale
        - Horizontal flip (does not destroy forensic artifacts)
        - Color jitter (robustness to lighting variations)
        - Random rotation (small angles)
        - Gaussian blur (robustness to quality variations)
        - ImageNet normalization

    Args:
        image_size: Target image size.

    Returns:
        Composition of training transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Composition of validation transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ForensicImageDataset(Dataset):
    """
    PyTorch Dataset for forensic image classification.

    Loads images from a structured directory and assigns class labels.
    Supports both binary and multi-class configurations.

    Attributes:
        samples: List of (image_path, label) tuples.
        transform: Image preprocessing transforms.
        class_to_idx: Mapping of class names to integer labels.
        idx_to_class: Reverse mapping of integer labels to class names.
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        max_samples_per_class: Optional[int] = None
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root_dir: Root directory containing class subdirectories.
            transform: Image transforms to apply.
            class_mapping: Custom class name to label mapping.
            max_samples_per_class: Maximum samples per class (for balancing).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or get_val_transforms()

        # Discover classes
        if class_mapping:
            self.class_to_idx = class_mapping
        else:
            classes = sorted([
                d.name for d in self.root_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Collect samples
        self.samples: List[Tuple[str, int]] = []
        self._load_samples(max_samples_per_class)

        logger.info(
            f"Dataset loaded: {len(self.samples)} samples, "
            f"{len(self.class_to_idx)} classes: {self.class_to_idx}"
        )

    def _load_samples(self, max_per_class: Optional[int]) -> None:
        """Discover and load image file paths with labels."""
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            class_samples = []
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    class_samples.append((str(file_path), label))

            if max_per_class and len(class_samples) > max_per_class:
                np.random.shuffle(class_samples)
                class_samples = class_samples[:max_per_class]

            self.samples.extend(class_samples)
            logger.info(f"  Class '{class_name}': {len(class_samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (transformed_image, label).
        """
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency class weights for loss balancing.

        Returns:
            Tensor of class weights.
        """
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels, minlength=len(self.class_to_idx))
        total = len(labels)
        weights = total / (len(self.class_to_idx) * class_counts + 1e-10)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Returns:
            Tensor of per-sample weights.
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for _, label in self.samples]
        return torch.FloatTensor(sample_weights)


class DataPipelineManager:
    """
    Manages the complete data pipeline: splitting, loading, and balancing.

    Provides methods for:
        - Train/Val/Test splitting
        - K-Fold cross-validation setup
        - DataLoader construction with optional oversampling
    """

    def __init__(
        self,
        dataset_root: str,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> None:
        """
        Initialize the data pipeline manager.

        Args:
            dataset_root: Root directory of the dataset.
            image_size: Target image size.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of parallel data loading workers.
            val_ratio: Fraction for validation set.
            test_ratio: Fraction for test set.
            seed: Random seed for reproducibility.
        """
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_dataloaders(
        self,
        use_oversampling: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create train, validation, and test DataLoaders.

        Args:
            use_oversampling: Whether to use weighted random sampling
                            to balance classes during training.

        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders.
        """
        # Load full dataset with validation transforms first (for splitting)
        full_dataset = ForensicImageDataset(
            self.dataset_root,
            transform=get_val_transforms(self.image_size)
        )

        # Stratified split
        labels = [label for _, label in full_dataset.samples]
        indices = list(range(len(full_dataset)))

        from sklearn.model_selection import train_test_split

        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_ratio,
            stratify=labels, random_state=self.seed
        )
        train_val_labels = [labels[i] for i in train_val_idx]
        adjusted_val_ratio = self.val_ratio / (1.0 - self.test_ratio)

        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=adjusted_val_ratio,
            stratify=train_val_labels, random_state=self.seed
        )

        # Create subset datasets with appropriate transforms
        train_dataset = _SubsetWithTransform(
            full_dataset, train_idx, get_train_transforms(self.image_size)
        )
        val_dataset = _SubsetWithTransform(
            full_dataset, val_idx, get_val_transforms(self.image_size)
        )
        test_dataset = _SubsetWithTransform(
            full_dataset, test_idx, get_val_transforms(self.image_size)
        )

        # Training sampler for class balancing
        train_sampler = None
        shuffle_train = True
        if use_oversampling and len(train_idx) > 0:
            train_labels = [labels[i] for i in train_idx]
            class_counts = np.bincount(train_labels)
            weights = 1.0 / (class_counts[train_labels] + 1e-10)
            train_sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(train_idx), replacement=True
            )
            shuffle_train = False

        loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle_train,
                sampler=train_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
        }

        logger.info(
            f"DataLoaders created: train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )

        return loaders

    def create_kfold_loaders(
        self, n_splits: int = 5
    ) -> List[Dict[str, DataLoader]]:
        """
        Create K-Fold cross-validation DataLoaders.

        Args:
            n_splits: Number of cross-validation folds.

        Returns:
            List of dictionaries, each with 'train' and 'val' DataLoaders.
        """
        full_dataset = ForensicImageDataset(
            self.dataset_root,
            transform=get_val_transforms(self.image_size)
        )

        labels = np.array([label for _, label in full_dataset.samples])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        fold_loaders = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(labels)), labels)):
            train_dataset = _SubsetWithTransform(
                full_dataset, train_idx.tolist(),
                get_train_transforms(self.image_size)
            )
            val_dataset = _SubsetWithTransform(
                full_dataset, val_idx.tolist(),
                get_val_transforms(self.image_size)
            )

            fold_loaders.append({
                'train': DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                ),
                'val': DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                ),
            })

            logger.info(
                f"Fold {fold + 1}/{n_splits}: "
                f"train={len(train_idx)}, val={len(val_idx)}"
            )

        return fold_loaders


class _SubsetWithTransform(Dataset):
    """
    Dataset subset wrapper that applies a specific transform.

    Used to apply different transforms (augmentation vs. none)
    to train and validation splits from the same base dataset.
    """

    def __init__(
        self,
        base_dataset: ForensicImageDataset,
        indices: List[int],
        transform: Callable
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        img_path, label = self.base_dataset.samples[real_idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label
