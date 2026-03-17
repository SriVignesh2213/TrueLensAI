"""
TrueLens AI - Training Pipeline
Complete training script for the AI-generated image detector.

Usage:
    python -m training.train --data_dir ./data --epochs 30 --batch_size 32

Dataset Structure:
    data/
    ├── train/
    │   ├── real/       # Real photographs
    │   └── ai_generated/  # AI-generated images
    ├── val/
    │   ├── real/
    │   └── ai_generated/
    └── test/
        ├── real/
        └── ai_generated/
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.detector import TrueLensDetector
from backend.utils.preprocessing import get_training_transform, get_inference_transform


def parse_args():
    parser = argparse.ArgumentParser(description='TrueLens AI - Training Pipeline')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate in classifier head')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pre-trained ImageNet weights')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone weights (train only classifier)')
    parser.add_argument('--unfreeze_epoch', type=int, default=5,
                        help='Epoch to unfreeze backbone (if frozen)')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to train on (auto/cpu/cuda)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computing device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    return device


def setup_data(data_dir, batch_size, num_workers):
    """Setup data loaders."""
    data_dir = Path(data_dir)
    
    # Transforms
    train_transform = get_training_transform()
    val_transform = get_inference_transform()
    
    # Datasets
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    datasets_dict = {}
    loaders_dict = {}
    
    if train_dir.exists():
        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
        datasets_dict['train'] = train_dataset
        loaders_dict['train'] = train_loader
        print(f"📁 Training set: {len(train_dataset)} images")
        print(f"   Classes: {train_dataset.classes}")
    else:
        print(f"⚠️  Training directory not found: {train_dir}")
    
    if val_dir.exists():
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        datasets_dict['val'] = val_dataset
        loaders_dict['val'] = val_loader
        print(f"📁 Validation set: {len(val_dataset)} images")
    
    if test_dir.exists():
        test_dataset = datasets.ImageFolder(str(test_dir), transform=val_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        datasets_dict['test'] = test_dataset
        loaders_dict['test'] = test_loader
        print(f"📁 Test set: {len(test_dataset)} images")
    
    return datasets_dict, loaders_dict


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Handle empty loader
    if len(loader) == 0:
        return 0.0, 0.0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress
        if (batch_idx + 1) % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} Acc: {acc:.2f}%")
    
    if len(loader) > 0:
        epoch_loss = running_loss / len(loader)
    else:
        epoch_loss = 0.0
        
    if total > 0:
        epoch_acc = 100.0 * correct / total
    else:
        epoch_acc = 0.0
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train(args):
    """Main training function."""
    print("=" * 60)
    print("  🔍 TrueLens AI - Training Pipeline")
    print("=" * 60)
    
    # Setup
    torch.manual_seed(args.seed)
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    datasets_dict, loaders = setup_data(args.data_dir, args.batch_size, args.num_workers)
    
    if 'train' not in loaders:
        print("❌ No training data found! Exiting.")
        return
    
    # Model
    print("\n🏗️  Building model...")
    model = TrueLensDetector(
        num_classes=2,
        pretrained=args.pretrained,
        dropout_rate=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("   ❄️  Backbone frozen (will unfreeze at epoch {args.unfreeze_epoch})")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Unfreeze backbone
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            print(f"\n🔓 Unfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Reset optimizer with all parameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate * 0.1,  # Lower LR for fine-tuning
                weight_decay=args.weight_decay
            )
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = 0.0, 0.0
        if 'val' in loaders:
            val_loss, val_acc, _, _ = evaluate(
                model, loaders['val'], criterion, device
            )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   LR: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"   ⭐ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc,
            }, ckpt_path)
            print(f"   💾 Checkpoint saved: {ckpt_path}")
        
        print("-" * 60)
    
    # Final evaluation on test set
    if 'test' in loaders:
        print("\n🧪 Final evaluation on test set...")
        # Load best model
        model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
        test_loss, test_acc, preds, labels = evaluate(
            model, loaders['test'], criterion, device
        )
        print(f"   Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Compute additional metrics
        from collections import Counter
        pred_counts = Counter(preds)
        label_counts = Counter(labels)
        print(f"   Prediction distribution: {dict(pred_counts)}")
        print(f"   Label distribution: {dict(label_counts)}")
    
    # Save final model for deployment
    final_model_path = output_dir / 'truelens_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📜 Training history saved: {history_path}")
    
    # Save training config
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    config['timestamp'] = datetime.now().isoformat()
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"  🏆 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_args()
    train(args)
