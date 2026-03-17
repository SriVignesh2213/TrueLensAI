"""
TrueLens AI — Ensemble Decision Fusion & Fraud Risk Scoring Engine

Combines outputs from all detection branches into a unified fraud assessment:
    - CNN Detector → ai_probability
    - Frequency Analyzer → spectral_anomaly_score
    - Metadata Analyzer → metadata_anomaly_score
    - Forgery Localizer → manipulation_score

Fusion Strategy:
    Weighted average with confidence-adaptive weighting.
    Branch weights are adjusted based on individual confidence levels.

Fraud Risk Score Formula:
    fraud_risk = w1 * ai_probability + w2 * manipulation_risk
               + w3 * metadata_anomaly + w4 * frequency_anomaly

    Where weights are dynamically adjusted based on per-branch confidence.

Risk Categories:
    - CRITICAL:  fraud_risk >= 0.85
    - HIGH:      fraud_risk >= 0.65
    - MEDIUM:    fraud_risk >= 0.40
    - LOW:       fraud_risk >= 0.20
    - MINIMAL:   fraud_risk < 0.20

Author: TrueLens AI Team
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import numpy as np
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class BranchResult:
    """
    Output from a single detection branch.

    Attributes:
        branch_name: Identifier for the detection branch.
        score: Raw anomaly/detection score [0, 1].
        confidence: Model confidence in this score [0, 1].
        details: Additional branch-specific details.
    """
    branch_name: str
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FraudAssessment:
    """
    Unified fraud assessment produced by the Decision Fusion Engine.

    This is the final output of the TrueLens AI analysis pipeline.

    Attributes:
        analysis_id: Unique identifier for this analysis.
        timestamp: ISO timestamp of analysis.
        ai_probability: Probability of image being AI-generated [0, 1].
        manipulation_risk: Risk of image manipulation [0, 1].
        metadata_anomaly: Whether metadata anomalies were detected.
        metadata_anomaly_score: Metadata anomaly severity [0, 1].
        frequency_anomaly_score: Frequency-domain anomaly score [0, 1].
        fraud_risk_score: Categorical risk level.
        fraud_risk_value: Numeric fraud risk [0, 1].
        confidence: Overall confidence in the assessment [0, 1].
        branch_results: Per-branch detailed results.
        heatmap_available: Whether heatmap data is available.
        suspicious_regions: Number of detected suspicious regions.
        recommendations: Actionable recommendations based on assessment.
    """
    analysis_id: str = ""
    timestamp: str = ""
    ai_probability: float = 0.0
    manipulation_risk: float = 0.0
    metadata_anomaly: bool = False
    metadata_anomaly_score: float = 0.0
    frequency_anomaly_score: float = 0.0
    fraud_risk_score: str = "MINIMAL"
    fraud_risk_value: float = 0.0
    confidence: float = 0.0
    branch_results: Dict[str, Any] = field(default_factory=dict)
    heatmap_available: bool = False
    suspicious_regions: int = 0
    recommendations: List[str] = field(default_factory=list)

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to structured API response format."""
        return {
            'analysis_id': self.analysis_id,
            'timestamp': self.timestamp,
            'ai_probability': round(self.ai_probability, 4),
            'manipulation_risk': round(self.manipulation_risk, 4),
            'metadata_anomaly': self.metadata_anomaly,
            'metadata_anomaly_score': round(self.metadata_anomaly_score, 4),
            'frequency_anomaly_score': round(self.frequency_anomaly_score, 4),
            'fraud_risk_score': self.fraud_risk_score,
            'fraud_risk_value': round(self.fraud_risk_value, 4),
            'confidence': round(self.confidence, 4),
            'heatmap_available': self.heatmap_available,
            'suspicious_regions': self.suspicious_regions,
            'recommendations': self.recommendations,
            'branch_results': self.branch_results,
        }


class DecisionFusionEngine:
    """
    Multi-branch ensemble decision fusion engine.

    Aggregates results from the CNN detector, frequency analyzer,
    metadata analyzer, and forgery localizer into a unified fraud
    risk assessment.

    Implements confidence-adaptive weighting: branches with higher
    confidence receive proportionally more influence.
    """

    # Base weights for each branch (before confidence adaptation)
    BASE_WEIGHTS = {
        'cnn_detector': 0.35,
        'frequency_analyzer': 0.20,
        'metadata_analyzer': 0.20,
        'forgery_localizer': 0.25,
    }

    # Risk thresholds
    RISK_THRESHOLDS = [
        (0.85, "CRITICAL"),
        (0.65, "HIGH"),
        (0.40, "MEDIUM"),
        (0.20, "LOW"),
        (0.00, "MINIMAL"),
    ]

    def __init__(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
        confidence_adaptation: bool = True
    ) -> None:
        """
        Initialize the Decision Fusion Engine.

        Args:
            custom_weights: Override base weights for each branch.
            confidence_adaptation: Whether to adapt weights based on confidence.
        """
        self.weights = custom_weights if custom_weights else self.BASE_WEIGHTS.copy()
        self.confidence_adaptation = confidence_adaptation
        logger.info(
            f"DecisionFusionEngine initialized: weights={self.weights}, "
            f"confidence_adaptation={confidence_adaptation}"
        )

    def fuse(
        self,
        cnn_result: Optional[BranchResult] = None,
        frequency_result: Optional[BranchResult] = None,
        metadata_result: Optional[BranchResult] = None,
        forgery_result: Optional[BranchResult] = None
    ) -> FraudAssessment:
        """
        Fuse results from all detection branches into a unified assessment.

        Args:
            cnn_result: CNN detector output.
            frequency_result: Frequency analyzer output.
            metadata_result: Metadata analyzer output.
            forgery_result: Forgery localizer output.

        Returns:
            FraudAssessment with unified fraud risk analysis.
        """
        assessment = FraudAssessment(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        # Collect available branch results
        branches: Dict[str, BranchResult] = {}
        if cnn_result:
            branches['cnn_detector'] = cnn_result
        if frequency_result:
            branches['frequency_analyzer'] = frequency_result
        if metadata_result:
            branches['metadata_analyzer'] = metadata_result
        if forgery_result:
            branches['forgery_localizer'] = forgery_result

        if not branches:
            logger.warning("No branch results provided for fusion")
            return assessment

        # Compute adaptive weights
        active_weights = self._compute_adaptive_weights(branches)

        # Weighted fraud risk score
        fraud_risk = sum(
            active_weights[name] * result.score
            for name, result in branches.items()
        )
        fraud_risk = min(1.0, max(0.0, fraud_risk))

        # Overall confidence (weighted average of branch confidences)
        overall_confidence = sum(
            active_weights[name] * result.confidence
            for name, result in branches.items()
        )

        # Populate assessment
        assessment.fraud_risk_value = fraud_risk
        assessment.fraud_risk_score = self._value_to_risk_level(fraud_risk)
        assessment.confidence = min(1.0, overall_confidence)

        # Map branch results to specific fields
        if cnn_result:
            assessment.ai_probability = cnn_result.score
        if forgery_result:
            assessment.manipulation_risk = forgery_result.score
            assessment.suspicious_regions = forgery_result.details.get('num_regions', 0)
            assessment.heatmap_available = forgery_result.details.get('heatmap_available', False)
        if metadata_result:
            assessment.metadata_anomaly_score = metadata_result.score
            assessment.metadata_anomaly = metadata_result.score >= 0.5
        if frequency_result:
            assessment.frequency_anomaly_score = frequency_result.score

        # Store detailed branch results
        assessment.branch_results = {
            name: {
                'score': round(r.score, 4),
                'confidence': round(r.confidence, 4),
                'weight': round(active_weights[name], 4),
                'details': r.details,
            }
            for name, r in branches.items()
        }

        # Generate recommendations
        assessment.recommendations = self._generate_recommendations(assessment)

        logger.info(
            f"Fusion complete: risk={assessment.fraud_risk_score} "
            f"({assessment.fraud_risk_value:.3f}), "
            f"confidence={assessment.confidence:.3f}"
        )

        return assessment

    def _compute_adaptive_weights(
        self, branches: Dict[str, BranchResult]
    ) -> Dict[str, float]:
        """
        Compute confidence-adaptive weights for available branches.

        Higher confidence branches receive proportionally more weight.
        Weights are re-normalized to sum to 1.0 among active branches.
        """
        if not self.confidence_adaptation:
            # Just re-normalize base weights for active branches
            total = sum(self.weights.get(name, 0) for name in branches)
            return {
                name: self.weights.get(name, 0) / max(total, 1e-10)
                for name in branches
            }

        # Confidence-adapted weights
        raw_weights = {}
        for name, result in branches.items():
            base = self.weights.get(name, 0.1)
            # Scale weight by confidence (sqrt to reduce extreme effects)
            adapted = base * np.sqrt(max(result.confidence, 0.01))
            raw_weights[name] = adapted

        total = sum(raw_weights.values())
        return {
            name: w / max(total, 1e-10)
            for name, w in raw_weights.items()
        }

    def _value_to_risk_level(self, value: float) -> str:
        """Convert numeric risk value to categorical risk level."""
        for threshold, level in self.RISK_THRESHOLDS:
            if value >= threshold:
                return level
        return "MINIMAL"

    @staticmethod
    def _generate_recommendations(assessment: FraudAssessment) -> List[str]:
        """Generate actionable recommendations based on the assessment."""
        recommendations = []

        if assessment.ai_probability > 0.8:
            recommendations.append(
                "HIGH AI-GENERATION PROBABILITY: This image shows strong indicators "
                "of being generated by an AI system. Treat with caution in any "
                "verification workflow."
            )
        elif assessment.ai_probability > 0.5:
            recommendations.append(
                "MODERATE AI INDICATORS: Some features suggest possible AI generation. "
                "Consider manual review or cross-referencing with additional sources."
            )

        if assessment.manipulation_risk > 0.7:
            recommendations.append(
                "SIGNIFICANT MANIPULATION DETECTED: Error Level Analysis reveals "
                "inconsistencies suggesting post-capture editing or splicing."
            )

        if assessment.metadata_anomaly:
            recommendations.append(
                "METADATA ANOMALIES: Image metadata shows irregularities. "
                "Missing camera signatures, software flags, or timestamp "
                "inconsistencies detected."
            )

        if assessment.frequency_anomaly_score > 0.6:
            recommendations.append(
                "SPECTRAL ARTIFACTS: Frequency analysis detected patterns "
                "consistent with synthetic image generation."
            )

        if not recommendations:
            recommendations.append(
                "LOW RISK: No significant indicators of AI generation or "
                "manipulation detected. Image appears authentic."
            )

        return recommendations
