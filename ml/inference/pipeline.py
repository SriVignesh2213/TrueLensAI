"""
TrueLens AI — Unified Inference Pipeline

Orchestrates all detection branches and produces a unified FraudAssessment.

Pipeline:
    Image Upload → Preprocessing → [CNN, FFT, Metadata, Forgery] → Fusion → Result

Author: TrueLens AI Team
License: MIT
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional
from pathlib import Path
import logging

from ml.models.efficientnet_detector import EfficientNetDetector
from ml.models.frequency_analyzer import SpectralFeatureExtractor, FrequencyClassifier
from ml.models.metadata_analyzer import MetadataForensicAnalyzer
from ml.models.forgery_localization import ForgeryLocalizer
from ml.models.decision_fusion import DecisionFusionEngine, BranchResult, FraudAssessment

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrueLensInferencePipeline:
    """Unified inference pipeline orchestrating all detection branches."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cnn_model = EfficientNetDetector(num_classes=2, pretrained=True)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.cnn_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            logger.info(f"Loaded trained model from {model_path}")
        self.cnn_model.to(self.device).eval()

        self.spectral_extractor = SpectralFeatureExtractor()
        self.freq_classifier = FrequencyClassifier()
        self.freq_classifier.to(self.device).eval()
        self.metadata_analyzer = MetadataForensicAnalyzer()
        self.forgery_localizer = ForgeryLocalizer()
        self.fusion_engine = DecisionFusionEngine()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        logger.info(f"Inference pipeline ready on {self.device}")

    def analyze(self, image_path: str) -> FraudAssessment:
        """Run full analysis pipeline on an image."""
        logger.info(f"Analyzing: {image_path}")

        # Branch 1: CNN Detection
        cnn_result = self._run_cnn(image_path)

        # Branch 2: Frequency Analysis
        freq_result = self._run_frequency(image_path)

        # Branch 3: Metadata Analysis
        meta_result = self._run_metadata(image_path)

        # Branch 4: Forgery Localization
        gradcam_heatmap = self._get_gradcam(image_path)
        forgery_result = self._run_forgery(image_path, gradcam_heatmap)

        # Fusion
        assessment = self.fusion_engine.fuse(
            cnn_result=cnn_result, frequency_result=freq_result,
            metadata_result=meta_result, forgery_result=forgery_result
        )
        return assessment

    def _run_cnn(self, image_path: str) -> BranchResult:
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        result = self.cnn_model.predict_with_confidence(tensor)
        ai_prob = float(result['probabilities'][0, 1])
        conf = float(result['confidence'][0])
        return BranchResult(branch_name='cnn_detector', score=ai_prob, confidence=conf, details={'prediction': int(result['prediction'][0])})

    def _run_frequency(self, image_path: str) -> BranchResult:
        try:
            image = np.array(Image.open(image_path).convert('L'))
            features = self.spectral_extractor.extract_spectral_features(image)
            feat_tensor = torch.FloatTensor(features['spectral_stats']).unsqueeze(0).to(self.device)
            result = self.freq_classifier.predict_probability(feat_tensor)
            return BranchResult(branch_name='frequency_analyzer', score=float(result['ai_probability'][0]), confidence=float(result['confidence'][0]), details={'high_freq_energy': features['high_freq_energy']})
        except Exception as e:
            logger.warning(f"Frequency analysis failed: {e}")
            return BranchResult(branch_name='frequency_analyzer', score=0.5, confidence=0.3)

    def _run_metadata(self, image_path: str) -> BranchResult:
        report = self.metadata_analyzer.analyze(image_path)
        return BranchResult(branch_name='metadata_analyzer', score=report.anomaly_score, confidence=0.85, details=report.to_dict())

    def _run_forgery(self, image_path: str, gradcam: Optional[np.ndarray] = None) -> BranchResult:
        result = self.forgery_localizer.localize(image_path, gradcam)
        return BranchResult(branch_name='forgery_localizer', score=result.manipulation_score, confidence=0.8, details={'num_regions': len(result.regions), 'heatmap_available': True, **result.to_dict()})

    def _get_gradcam(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            tensor.requires_grad = True
            gradcam = self.cnn_model.get_gradcam()
            return gradcam.generate(tensor).numpy()
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
            return None

    def get_heatmap_image(self, image_path: str) -> Optional[np.ndarray]:
        """Generate a heatmap overlay image for visualization."""
        import cv2
        try:
            heatmap = self._get_gradcam(image_path)
            if heatmap is None:
                return None
            original = cv2.imread(image_path)
            original = cv2.resize(original, (224, 224))
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
            return overlay
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
            return None
