"""
TrueLens AI - FastAPI Backend
Main application entry point providing RESTful API endpoints
for AI-generated image detection and analysis.
"""

import os
import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.detector import TrueLensDetector, load_detector
from backend.models.grad_cam import GradCAM
from backend.analysis.ela import ELAAnalyzer
from backend.analysis.frequency import FrequencyAnalyzer
from backend.analysis.metadata import MetadataAnalyzer
from backend.analysis.texture import TextureAnalyzer
from backend.utils.preprocessing import (
    load_image_from_bytes,
    preprocess_for_inference,
    get_image_info
)

# ============================================================
# Application Setup
# ============================================================

app = FastAPI(
    title="TrueLens AI",
    description="Deep Learning Framework for AI-Generated Image Detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# ============================================================
# Global State
# ============================================================

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")

# Model and analyzers (lazy loaded)
model = None
grad_cam = None
ela_analyzer = ELAAnalyzer(quality=90)
freq_analyzer = FrequencyAnalyzer()
meta_analyzer = MetadataAnalyzer()
texture_analyzer = TextureAnalyzer()
model_has_trained_weights = False

# Analysis history
analysis_history = []

# Model weights path
MODEL_WEIGHTS_PATH = Path(__file__).parent / "weights" / "truelens_model.pth"


def get_model():
    """Lazy load the detection model."""
    global model, grad_cam, model_has_trained_weights
    if model is None:
        weights_path = str(MODEL_WEIGHTS_PATH) if MODEL_WEIGHTS_PATH.exists() else None
        model_has_trained_weights = weights_path is not None
        if not model_has_trained_weights:
            print("⚠️  No trained weights found - DL predictions will be unreliable.")
            print("   Forensic analysis will be weighted more heavily.")
        model = load_detector(model_path=weights_path, device=DEVICE)
        try:
            grad_cam = GradCAM(model)
        except Exception as e:
            print(f"⚠️ Grad-CAM initialization warning: {e}")
            grad_cam = None
    return model


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>TrueLens AI - API is running</h1><p>Visit /docs for API documentation</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "version": "1.0.0"
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(True, description="Include Grad-CAM visualization"),
    include_ela: bool = Query(True, description="Include ELA analysis"),
    include_frequency: bool = Query(True, description="Include frequency analysis"),
    include_metadata: bool = Query(True, description="Include metadata analysis"),
    include_texture: bool = Query(True, description="Include texture/noise analysis")
):
    """
    Analyze an uploaded image for AI-generated content.
    
    This endpoint performs comprehensive analysis including:
    - Deep learning classification (EfficientNet-B0)
    - Error Level Analysis (ELA)
    - Frequency domain analysis
    - Metadata analysis
    - Grad-CAM explainability visualization
    
    Returns an overall authenticity score and detailed analysis results.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image (JPEG, PNG, WebP)."
            )
        
        # Read image bytes
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
        if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
        
        # Load image
        image = load_image_from_bytes(image_bytes)
        image_info = get_image_info(image)
        
        # ============================
        # 1. Deep Learning Classification
        # ============================
        detector = get_model()
        input_tensor = preprocess_for_inference(image).to(DEVICE)
        
        predicted_class, confidence, probabilities = detector.predict(input_tensor)
        
        class_labels = ["Real", "AI-Generated"]
        dl_result = {
            "prediction": class_labels[predicted_class],
            "confidence": round(confidence * 100, 2),
            "real_probability": round(probabilities[0] * 100, 2),
            "ai_probability": round(probabilities[1] * 100, 2),
        }
        
        # ============================
        # 2. Grad-CAM Visualization
        # ============================
        gradcam_result = None
        if include_gradcam and grad_cam is not None:
            try:
                overlay_b64, heatmap_b64, gc_class, gc_conf = grad_cam.generate_overlay(
                    input_tensor, image, target_class=predicted_class
                )
                gradcam_result = {
                    "overlay_image": overlay_b64,
                    "heatmap_image": heatmap_b64,
                    "target_class": class_labels[gc_class],
                    "available": True
                }
            except Exception as e:
                gradcam_result = {
                    "available": False,
                    "error": str(e)
                }
        
        # ============================
        # 3. Error Level Analysis
        # ============================
        ela_result = None
        if include_ela:
            try:
                ela_result = ela_analyzer.analyze(image)
            except Exception as e:
                ela_result = {"error": str(e), "manipulation_score": 0.5}
        
        # ============================
        # 4. Frequency Analysis
        # ============================
        freq_result = None
        if include_frequency:
            try:
                freq_result = freq_analyzer.analyze(image)
            except Exception as e:
                freq_result = {"error": str(e), "frequency_score": 0.5}
        
        # ============================
        # 5. Metadata Analysis
        # ============================
        meta_result = None
        if include_metadata:
            try:
                meta_result = meta_analyzer.analyze(image)
            except Exception as e:
                meta_result = {"error": str(e), "metadata_score": 0.5}
        
        # ============================
        # 6. Texture / Noise Analysis
        # ============================
        texture_result = None
        if include_texture:
            try:
                texture_result = texture_analyzer.analyze(image)
            except Exception as e:
                texture_result = {"error": str(e), "texture_score": 0.5}
        
        # ============================
        # 7. Compute Authenticity Score
        # ============================
        authenticity_score = compute_authenticity_score(
            dl_result, ela_result, freq_result, meta_result,
            texture_result, model_has_trained_weights
        )
        
        # Determine overall verdict
        if authenticity_score >= 70:
            verdict = "AUTHENTIC"
            verdict_detail = "This image appears to be a genuine, unmanipulated photograph."
        elif authenticity_score >= 45:
            verdict = "SUSPICIOUS"
            verdict_detail = "This image shows some indicators that warrant further examination."
        elif authenticity_score >= 25:
            verdict = "LIKELY AI-GENERATED"
            verdict_detail = "Multiple analysis methods indicate this image is likely AI-generated or significantly manipulated."
        else:
            verdict = "AI-GENERATED"
            verdict_detail = "Strong evidence indicates this image is AI-generated or heavily manipulated."
        
        processing_time = round(time.time() - start_time, 3)
        
        # Build response
        result = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "image_info": image_info,
            "authenticity_score": round(authenticity_score, 1),
            "verdict": verdict,
            "verdict_detail": verdict_detail,
            "deep_learning": dl_result,
            "gradcam": gradcam_result,
            "ela": ela_result,
            "frequency": freq_result,
            "metadata": meta_result,
            "texture": texture_result,
            "processing_time_seconds": processing_time,
            "model_trained": model_has_trained_weights,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store in history
        history_entry = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "authenticity_score": round(authenticity_score, 1),
            "verdict": verdict,
            "timestamp": result["timestamp"]
        }
        analysis_history.append(history_entry)
        if len(analysis_history) > 100:
            analysis_history.pop(0)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/history")
async def get_history():
    """Get recent analysis history."""
    return {
        "total": len(analysis_history),
        "analyses": list(reversed(analysis_history))
    }


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    detector = get_model()
    
    total_params = sum(p.numel() for p in detector.parameters())
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    
    return {
        "architecture": "EfficientNet-B0 + Custom Head",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_size": "224x224",
        "num_classes": 2,
        "class_labels": ["Real", "AI-Generated"],
        "device": str(DEVICE),
        "weights_loaded": MODEL_WEIGHTS_PATH.exists(),
        "framework": "PyTorch",
        "backbone": "EfficientNet-B0 (ImageNet pre-trained)"
    }


# ============================================================
# Scoring Logic
# ============================================================

def compute_authenticity_score(dl_result, ela_result, freq_result, meta_result,
                                texture_result=None, model_trained=False):
    """
    Compute a combined authenticity score from all analysis methods.
    
    Score ranges from 0 (definitely AI) to 100 (definitely real).
    
    When model is trained:
        - Deep Learning: 35%
        - ELA: 18%
        - Frequency: 17%
        - Metadata: 15%
        - Texture: 15%
    
    When model is NOT trained (no custom weights):
        - Deep Learning: 5% (mostly ignored — model is random)
        - ELA: 25%
        - Frequency: 25%
        - Metadata: 20%
        - Texture: 25%
    """
    scores = {}
    
    # 1. Deep Learning score
    dl_score = dl_result.get("real_probability", 50)
    scores["dl"] = dl_score
    
    # 2. ELA score (manipulation_score: 0=genuine, 1=manipulated)
    if ela_result and "manipulation_score" in ela_result:
        scores["ela"] = (1 - ela_result["manipulation_score"]) * 100
    
    # 3. Frequency score (frequency_score: 0=genuine, 1=AI)
    if freq_result and "frequency_score" in freq_result:
        scores["freq"] = (1 - freq_result["frequency_score"]) * 100
    
    # 4. Metadata score (metadata_score: 0=genuine, 1=suspicious)
    if meta_result and "metadata_score" in meta_result:
        scores["meta"] = (1 - meta_result["metadata_score"]) * 100
    
    # 5. Texture score (texture_score: 0=genuine, 1=AI)
    if texture_result and "texture_score" in texture_result:
        scores["texture"] = (1 - texture_result["texture_score"]) * 100
    
    # Set weights based on whether model has been trained
    if model_trained:
        weight_map = {
            "dl": 0.35,
            "ela": 0.18,
            "freq": 0.17,
            "meta": 0.15,
            "texture": 0.15
        }
    else:
        # Model is untrained — forensic methods dominate
        weight_map = {
            "dl": 0.05,    # Nearly ignored
            "ela": 0.25,
            "freq": 0.25,
            "meta": 0.20,
            "texture": 0.25
        }
    
    # Compute weighted score
    total_weight = 0
    weighted_sum = 0
    for key, score_val in scores.items():
        w = weight_map.get(key, 0.1)
        weighted_sum += score_val * w
        total_weight += w
    
    if total_weight > 0:
        weighted_score = weighted_sum / total_weight
    else:
        weighted_score = 50.0
    
    # Apply forensic consensus penalty:
    # If multiple forensic methods agree something is AI, push score lower
    forensic_scores = []
    for k in ["ela", "freq", "meta", "texture"]:
        if k in scores:
            forensic_scores.append(scores[k])
    
    if len(forensic_scores) >= 3:
        avg_forensic = sum(forensic_scores) / len(forensic_scores)
        # Count how many forensic methods say "likely AI" (score < 50)
        ai_votes = sum(1 for s in forensic_scores if s < 50)
        
        if ai_votes >= 3:
            # Strong consensus: push the score down significantly
            weighted_score = min(weighted_score, avg_forensic * 0.8)
        elif ai_votes >= 2:
            # Moderate consensus: blend toward forensic average
            weighted_score = weighted_score * 0.6 + avg_forensic * 0.4
    
    return max(0, min(100, weighted_score))


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("  🔍 TrueLens AI - Image Authenticity Detection System")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Frontend: {frontend_path}")
    print(f"  Model weights: {MODEL_WEIGHTS_PATH}")
    print("=" * 60)
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)]
    )
