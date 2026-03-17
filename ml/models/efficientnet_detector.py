"""
TrueLens AI — EfficientNet-based AI Image Detector

Transfer learning model using EfficientNet-B0 as backbone for binary
classification (Real vs AI-Generated). Supports Grad-CAM heatmap generation
for explainability.

Architecture:
    EfficientNet-B0 (frozen early layers)
    → Adaptive Average Pool
    → Dropout (0.3)
    → FC 1280→512 → ReLU → Dropout(0.2)
    → FC 512→2 (Real / AI-Generated)

Author: TrueLens AI Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visual
    explainability of CNN predictions.

    Generates heatmaps highlighting regions most influential in the
    model's classification decision.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Initialize Grad-CAM with model and target convolutional layer.

        Args:
            model: The neural network model.
            target_layer: The convolutional layer to compute Grad-CAM for.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_tensor: Preprocessed input image tensor [1, C, H, W].
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Heatmap tensor of shape [H, W] with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input dimensions
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )

        return cam.squeeze().cpu()


class EfficientNetDetector(nn.Module):
    """
    AI-Generated Image Detector based on EfficientNet-B0.

    Uses transfer learning with a custom classification head for binary
    detection of AI-generated vs real images. Supports optional multi-class
    classification for detection type categorization.

    Attributes:
        num_classes: Number of output classes (2 for binary, more for multi-class).
        backbone: EfficientNet-B0 feature extractor.
        classifier: Custom classification head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone_ratio: float = 0.7,
        dropout_rate: float = 0.3
    ) -> None:
        """
        Initialize the EfficientNet detector.

        Args:
            num_classes: Number of output classes.
            pretrained: Whether to use ImageNet-pretrained weights.
            freeze_backbone_ratio: Fraction of backbone layers to freeze (0.0 to 1.0).
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Extract the feature dimension from the original classifier
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Freeze early backbone layers for transfer learning stability
        self._freeze_backbone(freeze_backbone_ratio)

        # Custom classification head with regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier()

        logger.info(
            f"EfficientNetDetector initialized: classes={num_classes}, "
            f"frozen_ratio={freeze_backbone_ratio}, dropout={dropout_rate}"
        )

    def _freeze_backbone(self, ratio: float) -> None:
        """
        Freeze a proportion of backbone parameters for transfer learning.

        Args:
            ratio: Fraction of layers to freeze (from the input end).
        """
        params = list(self.backbone.parameters())
        freeze_count = int(len(params) * ratio)
        for i, param in enumerate(params):
            if i < freeze_count:
                param.requires_grad = False

        trainable = sum(p.requires_grad for p in self.backbone.parameters())
        total = len(params)
        logger.info(f"Backbone: {trainable}/{total} parameters trainable")

    def _init_classifier(self) -> None:
        """Initialize classifier weights using Kaiming initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and return predictions with confidence scores.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            Dictionary containing:
                - 'prediction': Predicted class indices [B]
                - 'confidence': Confidence scores [B]
                - 'probabilities': Class probability distribution [B, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities
        }

    def get_gradcam_layer(self) -> nn.Module:
        """
        Get the target layer for Grad-CAM visualization.

        Returns:
            The last convolutional block of EfficientNet features.
        """
        return self.backbone.features[-1]

    def get_gradcam(self) -> GradCAM:
        """
        Create a Grad-CAM instance for this model.

        Returns:
            GradCAM instance targeting the last conv layer.
        """
        return GradCAM(self, self.get_gradcam_layer())
