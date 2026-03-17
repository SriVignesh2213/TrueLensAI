"""
TrueLens AI — API Routes

FastAPI route definitions for the REST API.
Endpoints: POST /analyze-image, GET /analysis/{id}, GET /health, GET /history

Author: TrueLens AI Team
License: MIT
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.app.schemas.responses import (
    HealthResponse, AnalysisResponse, AnalysisHistoryResponse, ErrorResponse
)
from backend.app.services.analysis_service import AnalysisService
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
analysis_service = AnalysisService()
settings = get_settings()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(version=settings.APP_VERSION)


@router.post("/analyze-image", response_model=AnalysisResponse, tags=["Analysis"],
             responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}})
async def analyze_image(file: UploadFile = File(..., description="Image file to analyze")):
    """
    Analyze an image for AI generation, manipulation, and metadata anomalies.

    Runs the full TrueLens AI multi-branch detection pipeline and returns
    a unified fraud risk assessment with confidence scores.
    """
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # Validate file size
    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, detail=f"File too large. Max: {settings.MAX_FILE_SIZE_MB}MB")

    # Save uploaded file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, 'wb') as f:
        f.write(content)

    try:
        result = await analysis_service.analyze_image(file_path, file.filename)
        return AnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse, tags=["Analysis"],
            responses={404: {"model": ErrorResponse}})
async def get_analysis(analysis_id: str):
    """Retrieve a previous analysis result by its ID."""
    result = analysis_service.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(404, detail=f"Analysis {analysis_id} not found")
    return AnalysisResponse(**result)


@router.get("/history", response_model=AnalysisHistoryResponse, tags=["Analysis"])
async def get_analysis_history(limit: int = Query(50, ge=1, le=200)):
    """Get recent analysis history."""
    items = analysis_service.get_history(limit=limit)
    return AnalysisHistoryResponse(total=len(items), analyses=items)
