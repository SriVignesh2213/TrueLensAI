"""
TrueLens AI - Deep Learning Detector Model
EfficientNet-B0 based binary classifier for AI-generated image detection.
Uses transfer learning with a custom classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from torchvision import models


class TrueLensDetector(nn.Module):
    """
    AI-Generated Image Detector based on EfficientNet-B0.
    
    Architecture:
        - EfficientNet-B0 backbone (pre-trained on ImageNet)
        - Custom classification head with dropout regularization
        - Binary output: [Real, AI-Generated]
    
    The model leverages transfer learning to capture low-level forensic
    artifacts that distinguish real images from AI-generated ones.
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(TrueLensDetector, self).__init__()
        
        self.num_classes = num_classes
        
        if TIMM_AVAILABLE:
            # Use timm for better model support
            self.backbone = timm.create_model(
                'efficientnet_b0',
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg'
            )
            num_features = self.backbone.num_features
        else:
            # Fallback to torchvision
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            # Remove the classifier
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            num_features = 1280  # EfficientNet-B0 feature dim
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
        
        # Store the feature dimension for Grad-CAM
        self.num_features = num_features
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the classification head with Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_features(self, x):
        """
        Extract feature maps from the backbone.
        Used for Grad-CAM visualization.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Feature maps before global pooling
        """
        if TIMM_AVAILABLE:
            features = self.backbone.forward_features(x)
            return features
        else:
            # For torchvision, extract features before pooling
            for i, layer in enumerate(self.backbone):
                if i < len(self.backbone) - 1:
                    x = layer(x)
            return x
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        if TIMM_AVAILABLE:
            features = self.backbone(x)
        else:
            features = self.backbone(x)
            features = features.flatten(1)
        
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Probability tensor [B, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """
        Get prediction class and confidence.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        probs = self.predict_proba(x)
        confidence, predicted = torch.max(probs, dim=1)
        return predicted.item(), confidence.item(), probs.squeeze().tolist()


class TrueLensEnsemble(nn.Module):
    """
    Ensemble model that combines multiple detection approaches:
    1. EfficientNet-B0 for visual classification
    2. Frequency domain features
    3. ELA (Error Level Analysis) features
    
    The ensemble uses a learned fusion layer to combine predictions.
    """
    
    def __init__(self, pretrained=True):
        super(TrueLensEnsemble, self).__init__()
        
        # Primary visual classifier
        self.visual_detector = TrueLensDetector(
            num_classes=2, pretrained=pretrained
        )
        
        # Frequency domain branch (smaller network)
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32)
        )
        
        # ELA branch (smaller network)
        self.ela_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 + 32 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, image, freq_map, ela_map):
        """
        Forward pass with multi-modal inputs.
        
        Args:
            image: RGB image tensor [B, 3, 224, 224]
            freq_map: Frequency domain map [B, 1, H, W]
            ela_map: ELA image [B, 3, H, W]
            
        Returns:
            Logits tensor [B, 2]
        """
        # Visual features
        visual_logits = self.visual_detector(image)
        
        # Frequency features
        freq_features = self.freq_branch(freq_map)
        
        # ELA features
        ela_features = self.ela_branch(ela_map)
        
        # Fuse all features
        combined = torch.cat([visual_logits, freq_features, ela_features], dim=1)
        output = self.fusion(combined)
        
        return output


def load_detector(model_path=None, device='cpu'):
    """
    Load the TrueLens detector model.
    
    Args:
        model_path: Path to saved model weights (optional)
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    model = TrueLensDetector(num_classes=2, pretrained=True)
    
    if model_path:
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded model weights from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load weights from {model_path}: {e}")
            print("Using pre-trained ImageNet weights instead.")
    
    model = model.to(device)
    model.eval()
    return model
