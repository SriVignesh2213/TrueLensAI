"""
TrueLens AI — Sample Dataset Generator

Generates synthetic sample images to validate the training pipeline
end-to-end before using a real dataset.

Real images:  Natural-looking patterns (gradient noise, textures)
AI-generated: Synthetic patterns (grid artifacts, uniform noise)

This is for pipeline validation ONLY — replace with a real dataset
(e.g., CIFAKE, GenImage, or custom data) for production training.

Usage:
    python -m ml.data.generate_samples --output_dir ./dataset --num_per_class 100
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_real_style_image(size: int = 224, seed: int = 0) -> Image.Image:
    """
    Generate a synthetic 'real-looking' image with natural patterns.

    Simulates natural photography characteristics:
    - Smooth gradients (sky/landscape-like)
    - Gaussian noise (sensor noise)
    - Slight blur variation
    """
    rng = np.random.RandomState(seed)

    # Base gradient (simulates natural lighting)
    y_grad = np.linspace(0, 1, size).reshape(-1, 1)
    x_grad = np.linspace(0, 1, size).reshape(1, -1)

    # Random color channels with natural gradients
    r = (y_grad * rng.uniform(80, 200) + x_grad * rng.uniform(20, 80) +
         rng.normal(0, 15, (size, size)))
    g = (y_grad * rng.uniform(100, 220) + x_grad * rng.uniform(30, 90) +
         rng.normal(0, 12, (size, size)))
    b = (y_grad * rng.uniform(120, 240) + x_grad * rng.uniform(10, 60) +
         rng.normal(0, 18, (size, size)))

    img_array = np.stack([r, g, b], axis=2).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, 'RGB')

    # Add slight Gaussian blur (mimics lens softness)
    blur_radius = rng.uniform(0.5, 1.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Add random shapes (simulates objects)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 6)):
        x1, y1 = rng.randint(0, size, 2)
        x2, y2 = x1 + rng.randint(20, 80), y1 + rng.randint(20, 80)
        color = tuple(rng.randint(50, 200, 3).tolist())
        shape_type = rng.choice(['ellipse', 'rectangle'])
        if shape_type == 'ellipse':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=color)

    # Final slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    return img


def generate_ai_style_image(size: int = 224, seed: int = 0) -> Image.Image:
    """
    Generate a synthetic 'AI-generated-looking' image with artifacts.

    Simulates common AI generation artifacts:
    - Grid/periodic patterns (GAN checkerboard artifact)
    - Unnaturally smooth areas
    - Sharp color transitions
    - Repetitive textures
    """
    rng = np.random.RandomState(seed)

    # Start with smoother base (AI images tend to be very smooth)
    base = rng.uniform(60, 200, (size, size, 3))

    # Add periodic grid artifact (GAN signature)
    freq = rng.choice([4, 8, 16, 32])
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    grid = np.sin(2 * np.pi * xx / freq) * np.sin(2 * np.pi * yy / freq)
    grid = (grid * rng.uniform(10, 30))

    for c in range(3):
        base[:, :, c] += grid * rng.uniform(0.5, 1.5)

    # Add sharp color blocks (unnatural transitions)
    for _ in range(rng.randint(3, 8)):
        x1, y1 = rng.randint(0, size - 30, 2)
        w, h = rng.randint(15, 60, 2)
        color = rng.uniform(50, 220, 3)
        base[y1:y1+h, x1:x1+w, :] = color

    # Add subtle checkerboard pattern
    check_size = rng.choice([2, 4])
    checker = np.indices((size, size)).sum(axis=0) % (check_size * 2) < check_size
    base += checker[:, :, np.newaxis] * rng.uniform(3, 8)

    img_array = base.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, 'RGB')

    return img


def generate_dataset(
    output_dir: str = "./dataset",
    num_per_class: int = 100,
    image_size: int = 224
) -> None:
    """
    Generate a complete sample dataset for pipeline validation.

    Args:
        output_dir: Output directory (will create real/ and ai_generated/ subdirs).
        num_per_class: Number of images per class.
        image_size: Image dimensions (square).
    """
    real_dir = os.path.join(output_dir, "real")
    ai_dir = os.path.join(output_dir, "ai_generated")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    logger.info(f"Generating {num_per_class} 'real' images...")
    for i in range(num_per_class):
        img = generate_real_style_image(size=image_size, seed=i)
        img.save(os.path.join(real_dir, f"real_{i:04d}.jpg"), quality=92)

    logger.info(f"Generating {num_per_class} 'AI-generated' images...")
    for i in range(num_per_class):
        img = generate_ai_style_image(size=image_size, seed=i + 10000)
        img.save(os.path.join(ai_dir, f"ai_{i:04d}.jpg"), quality=92)

    logger.info(f"Dataset generated: {num_per_class * 2} total images in '{output_dir}'")
    logger.info(f"  ├── real/          ({num_per_class} images)")
    logger.info(f"  └── ai_generated/  ({num_per_class} images)")
    logger.info("")
    logger.info("⚠️  This is a SYNTHETIC dataset for pipeline validation only.")
    logger.info("   Replace with a real dataset (CIFAKE, GenImage, etc.) for production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample dataset for TrueLens AI")
    parser.add_argument('--output_dir', type=str, default='./dataset', help='Output directory')
    parser.add_argument('--num_per_class', type=int, default=100, help='Images per class')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.num_per_class, args.image_size)
