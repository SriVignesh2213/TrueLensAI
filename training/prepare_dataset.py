"""
TrueLens AI - Dataset Preparation Script
Downloads and organizes datasets for training the AI-generated image detector.

Supported datasets:
1. CIFAKE (Real vs AI-Generated)
2. Custom dataset from local directory
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset for TrueLens AI training')
    parser.add_argument('--source_real', type=str, required=True,
                        help='Directory containing real images')
    parser.add_argument('--source_ai', type=str, required=True,
                        help='Directory containing AI-generated images')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of training set (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation set (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum images per class (for debugging)')
    return parser.parse_args()


def get_image_files(directory):
    """Get all image files from a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    files = []
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if Path(filename).suffix.lower() in extensions:
                files.append(os.path.join(root, filename))
    
    return sorted(files)


def split_files(files, train_ratio, val_ratio, seed=42):
    """Split files into train/val/test sets."""
    random.seed(seed)
    random.shuffle(files)
    
    n = len(files)
    
    # Ensure at least one training image if we have any images
    train_end = int(n * train_ratio)
    if train_end == 0 and n > 0:
        train_end = 1
        
    val_end = int(n * (train_ratio + val_ratio))
    if val_end < train_end:
        val_end = train_end
        
    return {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }


def copy_files(files, dest_dir, class_name):
    """Copy files to destination directory."""
    if not files:
        return 0
        
    dest = Path(dest_dir) / class_name
    dest.mkdir(parents=True, exist_ok=True)
    
    for i, src_path in enumerate(files):
        ext = Path(src_path).suffix
        dst_path = dest / f"{class_name}_{i:06d}{ext}"
        shutil.copy2(src_path, dst_path)
    
    return len(files)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("  TrueLens AI - Dataset Preparation")
    print("=" * 60)
    
    # Collect files
    print(f"\n📁 Scanning real images: {args.source_real}")
    real_files = get_image_files(args.source_real)
    print(f"   Found {len(real_files)} real images")
    
    print(f"📁 Scanning AI-generated images: {args.source_ai}")
    ai_files = get_image_files(args.source_ai)
    print(f"   Found {len(ai_files)} AI-generated images")
    
    if not real_files or not ai_files:
        print("❌ Error: One or both directories are empty!")
        sys.exit(1)
    
    # Limit if requested
    if args.max_images:
        real_files = real_files[:args.max_images]
        ai_files = ai_files[:args.max_images]
        print(f"\n⚠️ Limited to {args.max_images} images per class")
    
    # Split datasets
    print(f"\n📊 Splitting with ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={1-args.train_ratio-args.val_ratio:.2f}")
    
    real_splits = split_files(real_files, args.train_ratio, args.val_ratio, args.seed)
    ai_splits = split_files(ai_files, args.train_ratio, args.val_ratio, args.seed)
    
    # Copy files
    print(f"\n💾 Copying files to {output_dir}...")
    total = 0
    
    for split_name in ['train', 'val', 'test']:
        real_count = copy_files(real_splits[split_name], output_dir / split_name, 'real')
        ai_count = copy_files(ai_splits[split_name], output_dir / split_name, 'ai_generated')
        total += real_count + ai_count
        print(f"   {split_name}: {real_count} real + {ai_count} AI-generated = {real_count + ai_count}")
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"   Total images: {total}")
    print(f"   Output directory: {output_dir}")
    print(f"\n📂 Structure:")
    print(f"   {output_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── real/")
    print(f"   │   └── ai_generated/")
    print(f"   ├── val/")
    print(f"   │   ├── real/")
    print(f"   │   └── ai_generated/")
    print(f"   └── test/")
    print(f"       ├── real/")
    print(f"       └── ai_generated/")


if __name__ == '__main__':
    main()
