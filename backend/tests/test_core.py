"""
TrueLens AI — Unit Tests

Tests for core ML components and API endpoints.

Author: TrueLens AI Team
License: MIT
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# ── Model Tests ──

class TestEfficientNetDetector:
    """Tests for EfficientNet-based AI detector."""

    def test_model_initialization(self):
        from ml.models.efficientnet_detector import EfficientNetDetector
        model = EfficientNetDetector(num_classes=2, pretrained=False)
        assert model.num_classes == 2

    def test_forward_pass_shape(self):
        from ml.models.efficientnet_detector import EfficientNetDetector
        model = EfficientNetDetector(num_classes=2, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 2)

    def test_predict_with_confidence(self):
        from ml.models.efficientnet_detector import EfficientNetDetector
        model = EfficientNetDetector(num_classes=2, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        result = model.predict_with_confidence(x)
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['confidence'][0] >= 0 and result['confidence'][0] <= 1

    def test_gradcam_layer(self):
        from ml.models.efficientnet_detector import EfficientNetDetector
        model = EfficientNetDetector(num_classes=2, pretrained=False)
        layer = model.get_gradcam_layer()
        assert layer is not None


class TestFrequencyAnalyzer:
    """Tests for frequency domain analysis."""

    def test_fft_magnitude(self):
        from ml.models.frequency_analyzer import SpectralFeatureExtractor
        extractor = SpectralFeatureExtractor()
        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        mag = extractor.compute_fft_magnitude(image.astype(np.float64) / 255.0)
        assert mag.shape == (224, 224)

    def test_spectral_features(self):
        from ml.models.frequency_analyzer import SpectralFeatureExtractor
        extractor = SpectralFeatureExtractor()
        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        features = extractor.extract_spectral_features(image)
        assert 'radial_profile' in features
        assert 'spectral_stats' in features
        assert len(features['spectral_stats']) == 12

    def test_frequency_classifier(self):
        from ml.models.frequency_analyzer import FrequencyClassifier
        clf = FrequencyClassifier()
        x = torch.randn(4, 12)
        out = clf(x)
        assert out.shape == (4, 2)


class TestMetadataAnalyzer:
    """Tests for metadata forensic analysis."""

    def test_analyzer_initialization(self):
        from ml.models.metadata_analyzer import MetadataForensicAnalyzer
        analyzer = MetadataForensicAnalyzer()
        assert analyzer is not None

    def test_score_to_risk_level(self):
        from ml.models.metadata_analyzer import MetadataForensicAnalyzer
        assert MetadataForensicAnalyzer._score_to_risk_level(0.9) == "CRITICAL"
        assert MetadataForensicAnalyzer._score_to_risk_level(0.6) == "HIGH"
        assert MetadataForensicAnalyzer._score_to_risk_level(0.3) == "MEDIUM"
        assert MetadataForensicAnalyzer._score_to_risk_level(0.1) == "LOW"


class TestDecisionFusion:
    """Tests for ensemble decision fusion."""

    def test_fusion_basic(self):
        from ml.models.decision_fusion import DecisionFusionEngine, BranchResult
        engine = DecisionFusionEngine()
        cnn = BranchResult(branch_name='cnn_detector', score=0.9, confidence=0.95)
        meta = BranchResult(branch_name='metadata_analyzer', score=0.7, confidence=0.85)
        result = engine.fuse(cnn_result=cnn, metadata_result=meta)
        assert result.fraud_risk_value > 0
        assert result.fraud_risk_score in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']

    def test_empty_fusion(self):
        from ml.models.decision_fusion import DecisionFusionEngine
        engine = DecisionFusionEngine()
        result = engine.fuse()
        assert result.fraud_risk_value == 0.0


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_binary_metrics(self):
        from ml.evaluation.metrics import MetricsCalculator
        calc = MetricsCalculator(num_classes=2)
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]])
        metrics = calc.compute_all(y_true, y_pred, y_prob)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1


# ── API Tests ──

class TestAPISchemas:
    """Tests for API response schemas."""

    def test_health_response(self):
        from backend.app.schemas.responses import HealthResponse
        resp = HealthResponse()
        assert resp.status == "healthy"

    def test_analysis_response(self):
        from backend.app.schemas.responses import AnalysisResponse
        resp = AnalysisResponse(
            analysis_id="test-123", timestamp="2026-01-01T00:00:00Z",
            ai_probability=0.9, manipulation_risk=0.7, metadata_anomaly=True,
            metadata_anomaly_score=0.65, frequency_anomaly_score=0.5,
            fraud_risk_score="HIGH", fraud_risk_value=0.78, confidence=0.92
        )
        assert resp.fraud_risk_score == "HIGH"
        assert resp.ai_probability == 0.9
