"""
TrueLens AI — Model Training Pipeline

Production-grade training loop with:
    - Mixed precision training (AMP)
    - Learning rate scheduling (Cosine Annealing with Warmup)
    - Early stopping with patience
    - Model checkpointing (best + periodic)
    - TensorBoard-compatible logging
    - Comprehensive metrics tracking
    - Gradient clipping for stability
    - Class-weighted loss for imbalanced data

Usage:
    python -m ml.training.train --data_dir ./dataset --epochs 50 --batch_size 32

Author: TrueLens AI Team
License: MIT
"""

import os
import sys
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import numpy as np

from ml.models.efficientnet_detector import EfficientNetDetector
from ml.data.dataset import DataPipelineManager
from ml.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class TrainingConfig:
    """
    Training hyperparameters and configuration.

    All parameters can be overridden via environment variables
    prefixed with TRUELENS_TRAIN_.
    """

    def __init__(
        self,
        data_dir: str = "./dataset",
        output_dir: str = "./checkpoints",
        num_classes: int = 2,
        image_size: int = 224,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        patience: int = 10,
        min_delta: float = 0.001,
        freeze_backbone_ratio: float = 0.7,
        dropout_rate: float = 0.3,
        use_amp: bool = True,
        gradient_clip_value: float = 1.0,
        num_workers: int = -1,
        save_every_n_epochs: int = 5,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.freeze_backbone_ratio = freeze_backbone_ratio
        self.dropout_rate = dropout_rate
        # Auto-disable AMP on CPU (only works with CUDA)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip_value = gradient_clip_value
        # Auto-detect num_workers: Windows has multiprocessing issues, default to 0
        if num_workers < 0:
            self.num_workers = 0 if platform.system() == 'Windows' else 4
        else:
            self.num_workers = num_workers
        self.save_every_n_epochs = save_every_n_epochs
        self.seed = seed

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs (patience).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        return self.should_stop


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup + cosine annealing.

    During warmup: LR increases linearly from 0 to base_lr.
    After warmup: LR follows cosine decay to min_lr.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update learning rate based on current epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(base_lr * factor, self.min_lr)


class ModelTrainer:
    """
    Production-grade model trainer for the EfficientNet detector.

    Manages the complete training lifecycle:
        1. Model initialization with transfer learning
        2. Data pipeline setup with augmentation
        3. Training loop with mixed precision
        4. Validation and metrics computation
        5. Checkpointing and early stopping
        6. Training history logging
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize the trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = EfficientNetDetector(
            num_classes=config.num_classes,
            pretrained=True,
            freeze_backbone_ratio=config.freeze_backbone_ratio,
            dropout_rate=config.dropout_rate,
        ).to(self.device)

        # Data pipeline
        self.data_manager = DataPipelineManager(
            dataset_root=config.data_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )

        # Training components (initialized in train())
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[WarmupCosineScheduler] = None
        self.scaler: Optional[GradScaler] = None
        self.criterion: Optional[nn.Module] = None
        self.early_stopping: Optional[EarlyStopping] = None

        # Metrics
        self.metrics_calc = MetricsCalculator(num_classes=config.num_classes)

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': [],
        }

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Config: {json.dumps(config.to_dict(), indent=2)}")

    def train(self) -> Dict[str, any]:
        """
        Execute the complete training pipeline.

        Returns:
            Dictionary with training results and best metrics.
        """
        # Setup data
        loaders = self.data_manager.create_dataloaders(use_oversampling=True)
        train_loader = loaders['train']
        val_loader = loaders['val']

        # Optimizer with differential learning rates
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        classifier_params = list(self.model.classifier.parameters())

        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.config.learning_rate},
        ], weight_decay=self.config.weight_decay)

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.config.warmup_epochs,
            total_epochs=self.config.epochs,
        )

        # Loss function with class weights
        try:
            class_weights = loaders['train'].dataset.base_dataset.get_class_weights()
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device), label_smoothing=0.1
            )
        except Exception:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Mixed precision (CUDA only)
        if self.config.use_amp and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0

        logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Update learning rate
            self.scheduler.step(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            val_loss, val_acc, val_metrics = self._validate_epoch(val_loader, epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=False)

            # Early stopping check
            if self.early_stopping(val_loss):
                logger.info(f"Training stopped early at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time

        # Final evaluation on test set
        test_loader = loaders['test']
        test_metrics = self._evaluate(test_loader)

        # Save training history
        self._save_history()

        results = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'total_epochs_trained': epoch + 1,
            'training_time_seconds': total_time,
            'test_metrics': test_metrics,
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"Best Validation: Epoch {best_epoch}, Loss={best_val_loss:.4f}, Acc={best_val_acc:.4f}")
        logger.info(f"Test Metrics: {json.dumps(test_metrics, indent=2)}")
        logger.info(f"Total Time: {total_time:.1f}s")
        logger.info(f"{'='*60}")

        return results

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.config.use_amp and self.scaler:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_value
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_value
                )
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def _validate_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Tuple[float, float, Dict]:
        """Run one validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = probs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        metrics = self.metrics_calc.compute_all(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )

        return epoch_loss, epoch_acc, metrics

    def _evaluate(self, loader: DataLoader) -> Dict:
        """Full evaluation on a DataLoader."""
        _, _, metrics = self._validate_epoch(loader, -1)
        return metrics

    def _save_checkpoint(
        self, epoch: int, val_loss: float, val_acc: float, is_best: bool
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config.to_dict(),
        }

        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth'

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="TrueLens AI - Model Training")
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        ]
    )

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_classes=args.num_classes,
        image_size=args.image_size,
        seed=args.seed,
    )

    trainer = ModelTrainer(config)
    results = trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
