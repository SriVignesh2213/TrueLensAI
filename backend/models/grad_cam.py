"""
TrueLens AI - Grad-CAM Explainability Module
Implements Gradient-weighted Class Activation Mapping for visual explanations
of model predictions, highlighting regions that influenced the detection decision.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Produces a coarse localization map highlighting the important regions
    in the image for predicting the target class. This helps explain
    WHY the model classified an image as real or AI-generated.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from 
    Deep Networks via Gradient-based Localization", ICCV 2017.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The TrueLensDetector model
            target_layer: The target convolutional layer for Grad-CAM.
                         If None, uses the last convolutional layer.
        """
        self.model = model
        self.model.eval()
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks on target layer
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        self.target_layer = target_layer
        self._register_hooks()
    
    def _find_target_layer(self):
        """Find the last convolutional layer in the backbone."""
        target = None
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        if target is None:
            # Fallback: use the backbone itself
            target = self.model.backbone
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input tensor [1, 3, 224, 224]
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            heatmap: NumPy array [H, W] with values in [0, 1]
            predicted_class: The predicted class index
            confidence: The prediction confidence
        """
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        if gradients is None or activations is None:
            # Return uniform heatmap if hooks didn't capture
            return np.ones((224, 224)) * 0.5, target_class, confidence
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = self._resize_heatmap(cam, (224, 224))
        
        return cam, target_class, confidence
    
    def _resize_heatmap(self, heatmap, target_size):
        """Resize heatmap to target size."""
        if CV2_AVAILABLE:
            heatmap = cv2.resize(heatmap, target_size)
        else:
            # Fallback using PIL
            heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize(target_size, Image.BILINEAR)
            heatmap = np.array(heatmap_img).astype(np.float32) / 255.0
        return heatmap
    
    def generate_overlay(self, input_tensor, original_image, target_class=None, alpha=0.5):
        """
        Generate Grad-CAM heatmap overlaid on the original image.
        
        Args:
            input_tensor: Preprocessed input tensor [1, 3, 224, 224]
            original_image: Original PIL Image
            target_class: Target class index
            alpha: Opacity of the heatmap overlay
            
        Returns:
            overlay_base64: Base64 encoded overlay image
            heatmap_base64: Base64 encoded heatmap image  
            predicted_class: Predicted class
            confidence: Prediction confidence
        """
        heatmap, pred_class, confidence = self.generate(input_tensor, target_class)
        
        # Resize original image to match heatmap
        orig_resized = original_image.resize((224, 224))
        orig_array = np.array(orig_resized)
        
        # Create colored heatmap
        heatmap_colored = self._apply_colormap(heatmap)
        
        # Create overlay
        overlay = (orig_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        
        # Convert to base64
        overlay_b64 = self._array_to_base64(overlay)
        heatmap_b64 = self._array_to_base64(heatmap_colored)
        
        return overlay_b64, heatmap_b64, pred_class, confidence
    
    def _apply_colormap(self, heatmap):
        """Apply a JET-like colormap to the heatmap."""
        if CV2_AVAILABLE:
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            return colored
        else:
            # Manual JET-like colormap
            h, w = heatmap.shape
            colored = np.zeros((h, w, 3), dtype=np.uint8)
            
            for i in range(h):
                for j in range(w):
                    val = heatmap[i, j]
                    r, g, b = self._jet_color(val)
                    colored[i, j] = [r, g, b]
            
            return colored
    
    @staticmethod
    def _jet_color(value):
        """Convert a value [0,1] to JET colormap RGB."""
        # Simplified JET colormap
        if value < 0.125:
            r, g, b = 0, 0, 128 + value * 1024
        elif value < 0.375:
            r, g, b = 0, (value - 0.125) * 1024, 255
        elif value < 0.625:
            r, g, b = (value - 0.375) * 1024, 255, 255 - (value - 0.375) * 1024
        elif value < 0.875:
            r, g, b = 255, 255 - (value - 0.625) * 1024, 0
        else:
            r, g, b = 255 - (value - 0.875) * 1024, 0, 0
        
        return int(min(255, max(0, r))), int(min(255, max(0, g))), int(min(255, max(0, b)))
    
    @staticmethod
    def _array_to_base64(array):
        """Convert NumPy array to base64 encoded PNG string."""
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
