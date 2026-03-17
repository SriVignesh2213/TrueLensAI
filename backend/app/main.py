"""
TrueLens AI — FastAPI Application Entry Point

Production-grade FastAPI application with CORS, logging, and error handling.

Usage:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

Author: TrueLens AI Team
License: MIT
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import router
from backend.app.core.config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting...")
    yield
    logger.info(f"🛑 {settings.APP_NAME} shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Multi-Layer Digital Media Forensics & Fraud Intelligence Platform. "
        "Detects AI-generated images, manipulation, metadata anomalies, and "
        "frequency-domain artifacts with explainable heatmaps."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={
        "error": "Internal Server Error",
        "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
        "status_code": 500,
    })
