"""
TrueLens AI - Image Preprocessing Utilities
Handles image loading, normalization, and transformation for model inference.
"""

import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard input size for EfficientNet-B0
INPUT_SIZE = 224


def get_inference_transform():
    """
    Returns the standard inference transform pipeline.
    Resize -> CenterCrop -> ToTensor -> Normalize
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform():
    """
    Returns the training transform pipeline with data augmentation.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load a PIL Image from raw bytes.
    
    Args:
        image_bytes: Raw image file bytes
        
    Returns:
        PIL Image in RGB mode
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def preprocess_for_inference(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.
    
    Args:
        image: PIL Image in RGB mode
        
    Returns:
        Preprocessed tensor with batch dimension [1, 3, 224, 224]
    """
    transform = get_inference_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor back to pixel values [0, 255].
    
    Args:
        tensor: Normalized tensor [C, H, W]
        
    Returns:
        NumPy array [H, W, C] with values in [0, 255]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    tensor = tensor.clone()
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to HWC format
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def get_image_info(image: Image.Image) -> dict:
    """
    Extract basic image information.
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with image metadata
    """
    info = {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": getattr(image, 'format', 'Unknown'),
        "size_pixels": image.width * image.height,
    }
    
    # Extract EXIF data if available
    exif_data = {}
    try:
        from PIL.ExifTags import TAGS
        raw_exif = image._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                try:
                    exif_data[str(tag_name)] = str(value)
                except Exception:
                    pass
    except Exception:
        pass
    
    info["exif"] = exif_data
    info["has_exif"] = len(exif_data) > 0
    
    return info
