"""
TrueLens AI — Pydantic Response Schemas

Defines structured API response models for type safety and documentation.

Author: TrueLens AI Team
License: MIT
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    service: str = "TrueLens AI"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class AnalysisResponse(BaseModel):
    """Full image analysis response."""
    analysis_id: str
    timestamp: str
    ai_probability: float = Field(ge=0, le=1, description="Probability of AI generation")
    manipulation_risk: float = Field(ge=0, le=1, description="Manipulation risk score")
    metadata_anomaly: bool = Field(description="Whether metadata anomalies were detected")
    metadata_anomaly_score: float = Field(ge=0, le=1)
    frequency_anomaly_score: float = Field(ge=0, le=1)
    fraud_risk_score: str = Field(description="CRITICAL, HIGH, MEDIUM, LOW, or MINIMAL")
    fraud_risk_value: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    heatmap_available: bool = False
    suspicious_regions: int = 0
    recommendations: List[str] = []
    branch_results: Dict[str, Any] = {}

    model_config = {"json_schema_extra": {"example": {
        "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "timestamp": "2026-02-23T15:30:00Z",
        "ai_probability": 0.92,
        "manipulation_risk": 0.73,
        "metadata_anomaly": True,
        "metadata_anomaly_score": 0.65,
        "frequency_anomaly_score": 0.58,
        "fraud_risk_score": "HIGH",
        "fraud_risk_value": 0.78,
        "confidence": 0.94,
        "heatmap_available": True,
        "suspicious_regions": 2,
        "recommendations": ["HIGH AI-GENERATION PROBABILITY"],
        "branch_results": {}
    }}}


class AnalysisHistoryItem(BaseModel):
    """Summary item for analysis history."""
    analysis_id: str
    timestamp: str
    filename: str
    fraud_risk_score: str
    fraud_risk_value: float
    ai_probability: float


class AnalysisHistoryResponse(BaseModel):
    """Analysis history list response."""
    total: int
    analyses: List[AnalysisHistoryItem]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    status_code: int
