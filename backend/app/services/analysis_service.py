"""
TrueLens AI — Analysis Service

Core business logic for image analysis orchestration.
Manages the inference pipeline, result storage, and heatmap generation.

Author: TrueLens AI Team
License: MIT
"""

import os
import json
import shutil
import base64
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


class AnalysisService:
    """Manages image analysis lifecycle: upload, analyze, store, retrieve."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pipeline = None
        self._results_store: Dict[str, Dict] = {}

        os.makedirs(self.settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.settings.RESULTS_DIR, exist_ok=True)
        logger.info("AnalysisService initialized")

    @property
    def pipeline(self):
        """Lazy-load the inference pipeline."""
        if self._pipeline is None:
            from ml.inference.pipeline import TrueLensInferencePipeline
            self._pipeline = TrueLensInferencePipeline(
                model_path=self.settings.MODEL_PATH,
                device=self.settings.DEVICE if self.settings.DEVICE != "auto" else None,
            )
        return self._pipeline

    async def analyze_image(self, file_path: str, filename: str) -> Dict:
        """Run full analysis pipeline on an uploaded image."""
        logger.info(f"Starting analysis for: {filename}")

        assessment = self.pipeline.analyze(file_path)
        result = assessment.to_api_response()
        result['filename'] = filename

        # Generate heatmap
        heatmap_b64 = self._generate_heatmap_b64(file_path)
        if heatmap_b64:
            result['heatmap_base64'] = heatmap_b64
            result['heatmap_available'] = True

        # Store result
        self._results_store[result['analysis_id']] = result
        self._persist_result(result)

        logger.info(f"Analysis complete: {result['analysis_id']} — {result['fraud_risk_score']}")
        return result

    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Retrieve a stored analysis result by ID."""
        if analysis_id in self._results_store:
            return self._results_store[analysis_id]
        return self._load_persisted_result(analysis_id)

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get analysis history summaries."""
        results = sorted(self._results_store.values(), key=lambda r: r.get('timestamp', ''), reverse=True)
        return [{
            'analysis_id': r['analysis_id'],
            'timestamp': r['timestamp'],
            'filename': r.get('filename', 'unknown'),
            'fraud_risk_score': r['fraud_risk_score'],
            'fraud_risk_value': r['fraud_risk_value'],
            'ai_probability': r['ai_probability'],
        } for r in results[:limit]]

    def _generate_heatmap_b64(self, image_path: str) -> Optional[str]:
        try:
            overlay = self.pipeline.get_heatmap_image(image_path)
            if overlay is not None:
                import cv2
                _, buffer = cv2.imencode('.png', overlay)
                return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
        return None

    def _persist_result(self, result: Dict) -> None:
        try:
            path = Path(self.settings.RESULTS_DIR) / f"{result['analysis_id']}.json"
            safe_result = {k: v for k, v in result.items() if k != 'heatmap_base64'}
            with open(path, 'w') as f:
                json.dump(safe_result, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to persist result: {e}")

    def _load_persisted_result(self, analysis_id: str) -> Optional[Dict]:
        try:
            path = Path(self.settings.RESULTS_DIR) / f"{analysis_id}.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load result {analysis_id}: {e}")
        return None
