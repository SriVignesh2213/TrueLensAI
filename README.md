<div align="center">

# 🔍 TrueLens AI

### Multi-Layer Digital Media Forensics & Fraud Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Enterprise-grade AI-powered image authenticity verification combining CNN classification, frequency-domain analysis, EXIF metadata forensics, and Error Level Analysis with explainable heatmaps.**

[🚀 Quick Start](#-quick-start) · [📖 API Docs](#-api-documentation) · [🧠 Architecture](#-system-architecture) · [🔬 ML Pipeline](#-ml-pipeline)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [ML Pipeline](#-ml-pipeline)
- [API Documentation](#-api-documentation)
- [Frontend Dashboard](#-frontend-dashboard)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Ethical Considerations](#-ethical-considerations)
- [Future Roadmap](#-future-roadmap)

---

## Overview

TrueLens AI is a production-grade digital media forensics platform that combines multiple AI and signal processing techniques to assess the authenticity of images. It is designed for organizations that need reliable, explainable, and scalable image verification — from financial institutions combating identity fraud to social media platforms moderating synthetic content.

### 🎯 Industry Applications

| Sector | Use Case |
|--------|----------|
| **FinTech** | KYC document fraud prevention, deepfake detection in identity verification |
| **Social Media** | AI-generated content labeling, misinformation mitigation |
| **E-Commerce** | Product image authenticity, dispute evidence verification |
| **Insurance** | Claims fraud detection, damage photo manipulation detection |
| **Law Enforcement** | Digital evidence integrity verification, forensic investigation support |

---

## ✨ Key Features

- **🧠 CNN-Based AI Detection** — EfficientNet-B0 with transfer learning for binary/multi-class classification of AI-generated vs. real images
- **📊 Frequency Domain Analysis** — 2D FFT spectral analysis detecting GAN grid artifacts and diffusion model noise patterns
- **🏷️ EXIF Metadata Forensics** — Deep metadata inspection detecting missing camera signatures, software manipulation flags, timestamp anomalies
- **🔍 Forgery Localization** — Error Level Analysis (ELA) + Grad-CAM for pixel-level suspicious region detection with bounding boxes
- **🎯 Ensemble Decision Fusion** — Confidence-adaptive weighted fusion across all detection branches
- **📈 Fraud Risk Scoring** — Unified 0-100% risk score with CRITICAL/HIGH/MEDIUM/LOW/MINIMAL categorization
- **🗺️ Explainable Heatmaps** — Grad-CAM visual overlays showing exactly why the model flagged an image
- **🌐 REST API** — Async FastAPI backend with structured JSON responses for enterprise integration
- **💻 Interactive Dashboard** — React + Tailwind CSS dashboard with risk meter, score breakdown, and analysis history

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │  React Dashboard │  │  REST API Consumers (Enterprise)     │ │
│  │  (Tailwind CSS)  │  │  POST /api/v1/analyze-image          │ │
│  └────────┬─────────┘  └────────────────┬─────────────────────┘ │
└───────────┼─────────────────────────────┼───────────────────────┘
            │                             │
┌───────────▼─────────────────────────────▼───────────────────────┐
│                      API GATEWAY (FastAPI)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ /analyze    │  │ /analysis/id │  │ /health  /history      │ │
│  │ (POST)      │  │ (GET)        │  │ (GET)                  │ │
│  └──────┬──────┘  └──────────────┘  └────────────────────────┘ │
└─────────┼──────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                            │
│  Image Validation → Resize → Normalize → Format Conversion     │
└─────────┬──────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│              MULTI-BRANCH DETECTION ENGINE                      │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────┐│
│  │ CNN Detector  │ │ FFT Analyzer │ │ Metadata   │ │ Forgery  ││
│  │ EfficientNet  │ │ Spectral     │ │ EXIF       │ │ ELA +    ││
│  │ + Grad-CAM    │ │ Features     │ │ Forensics  │ │ Grad-CAM ││
│  │              │ │ + MLP        │ │            │ │          ││
│  │ ai_prob      │ │ freq_score   │ │ meta_score │ │ manip_   ││
│  │ confidence   │ │ confidence   │ │ anomalies  │ │ score    ││
│  └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └────┬─────┘│
└─────────┼────────────────┼───────────────┼──────────────┼──────┘
          │                │               │              │
┌─────────▼────────────────▼───────────────▼──────────────▼──────┐
│                 DECISION FUSION ENGINE                          │
│  Confidence-Adaptive Weighted Ensemble                         │
│  w1·CNN + w2·FFT + w3·Metadata + w4·Forgery → fraud_risk      │
└─────────┬──────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│                 FRAUD RISK SCORING                              │
│  0.0 ──── MINIMAL ── LOW ── MEDIUM ── HIGH ── CRITICAL ── 1.0 │
│  + Actionable Recommendations                                  │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Upload** — Image submitted via dashboard or API endpoint
2. **Validate** — File type, size, and integrity checks
3. **Preprocess** — Resize, normalize, format conversion for each branch
4. **Detect** — Four parallel detection branches analyze the image
5. **Fuse** — Confidence-adaptive weighted ensemble combines all branch scores
6. **Score** — Unified fraud risk score with categorical risk level
7. **Explain** — Grad-CAM heatmaps and ELA maps provide visual explanations
8. **Respond** — Structured JSON response with all scores and recommendations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) Docker & Docker Compose

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/truelens-ai.git
cd truelens-ai

# 2. Backend setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# 3. Start the backend
uvicorn backend.app.main:app --reload --port 8000

# 4. Frontend setup (new terminal)
cd frontend
npm install
npm run dev

# 5. Open http://localhost:3000
```

### Docker Deployment

```bash
docker-compose up --build
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:3000
```

---

## 🔬 ML Pipeline

### Model Architecture

**Primary Detector:** EfficientNet-B0 with transfer learning
- ImageNet-pretrained backbone (70% frozen layers)
- Custom classification head: 1280 → 512 → 128 → 2
- Dropout (0.3) + BatchNorm for regularization
- Grad-CAM on final convolutional block

### Training Pipeline

```bash
python -m ml.training.train \
    --data_dir ./dataset \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4
```

### Dataset Preparation

Organize your dataset as:
```
dataset/
├── real/          # Authentic camera images
│   ├── img_001.jpg
│   └── ...
└── ai_generated/  # AI-generated images
    ├── img_001.jpg
    └── ...
```

### Overfitting Prevention Strategy

| Technique | Implementation |
|-----------|---------------|
| Data Augmentation | Random crop, flip, color jitter, Gaussian blur, random erasing |
| Dropout | 0.3 (head), 0.2 (intermediate) |
| Label Smoothing | 0.1 |
| Early Stopping | Patience = 10 epochs |
| Weight Decay | L2 = 1e-4 |
| Learning Rate | Cosine annealing with 5-epoch warmup |
| Gradient Clipping | Max norm = 1.0 |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | True positive rate (minimize false accusations) |
| Recall | Detection rate (minimize missed AI images) |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under receiver operating characteristic |
| FPR | False positive rate (critical for forensics) |

---

## 📖 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze-image` | Analyze an image for authenticity |
| `GET` | `/api/v1/analysis/{id}` | Retrieve previous analysis |
| `GET` | `/api/v1/history` | Get analysis history |
| `GET` | `/api/v1/health` | Health check |

### `POST /api/v1/analyze-image`

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2026-02-23T15:30:00Z",
  "ai_probability": 0.92,
  "manipulation_risk": 0.73,
  "metadata_anomaly": true,
  "metadata_anomaly_score": 0.65,
  "frequency_anomaly_score": 0.58,
  "fraud_risk_score": "HIGH",
  "fraud_risk_value": 0.78,
  "confidence": 0.94,
  "heatmap_available": true,
  "suspicious_regions": 2,
  "recommendations": [
    "HIGH AI-GENERATION PROBABILITY: This image shows strong indicators of being generated by an AI system."
  ],
  "branch_results": {
    "cnn_detector": { "score": 0.92, "confidence": 0.95, "weight": 0.38 },
    "frequency_analyzer": { "score": 0.58, "confidence": 0.72, "weight": 0.18 },
    "metadata_analyzer": { "score": 0.65, "confidence": 0.85, "weight": 0.20 },
    "forgery_localizer": { "score": 0.73, "confidence": 0.80, "weight": 0.24 }
  }
}
```

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## 💻 Frontend Dashboard

The React dashboard provides:

- **Drag & Drop Upload** — Upload images with real-time preview
- **Risk Meter** — Animated SVG semicircle gauge with color-coded risk levels
- **Detection Breakdown** — Per-branch score bars with confidence values
- **Heatmap Overlay** — Toggle Grad-CAM forensic heatmaps on the analyzed image
- **Analysis History** — Scrollable list of previous analyses with quick access
- **Responsive Design** — Mobile-first, glassmorphism aesthetic

---

## 🚢 Deployment

### Docker (Recommended)

```bash
docker-compose up -d --build
```

### Cloud Deployment

**AWS ECS / GCP Cloud Run:**
1. Build and push Docker images to ECR/GCR
2. Create task definitions / services
3. Configure ALB for routing
4. Set environment variables from Secrets Manager

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./checkpoints/best_model.pth` | Path to trained model weights |
| `DEVICE` | `auto` | PyTorch device (auto/cpu/cuda) |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload file size |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 📁 Project Structure

```
TrueLensAI/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints
│   │   ├── core/config.py         # Pydantic settings
│   │   ├── schemas/responses.py   # Response models
│   │   ├── services/              # Business logic
│   │   └── main.py                # App entry point
│   └── tests/test_core.py         # Unit tests
├── ml/
│   ├── models/
│   │   ├── efficientnet_detector.py  # CNN + Grad-CAM
│   │   ├── frequency_analyzer.py     # FFT spectral analysis
│   │   ├── metadata_analyzer.py      # EXIF forensics
│   │   ├── forgery_localization.py   # ELA + region detection
│   │   └── decision_fusion.py        # Ensemble fusion
│   ├── training/train.py          # Training pipeline
│   ├── inference/pipeline.py      # Unified inference
│   ├── evaluation/metrics.py      # Eval metrics
│   └── data/dataset.py            # Data pipeline
├── frontend/
│   ├── src/
│   │   ├── components/            # React components
│   │   ├── utils/api.js           # API client
│   │   ├── App.jsx                # Main dashboard
│   │   └── main.jsx               # Entry point
│   └── index.html
├── Dockerfile                     # Backend Docker
├── Dockerfile.frontend            # Frontend Docker
├── docker-compose.yml             # Full-stack orchestration
├── requirements.txt               # Python dependencies
├── .env                           # Configuration
└── README.md                      # This file
```

---

## ⚖️ Ethical Considerations

### Responsible Use

TrueLens AI is designed as a **decision-support tool**, not an autonomous judge. All results should be interpreted by qualified professionals.

### Bias & Limitations

| Limitation | Mitigation |
|-----------|------------|
| **Training data bias** | Models may underperform on image types not in training data. Use diverse, representative datasets. |
| **False positives** | High-quality AI images may be classified as real. Use multi-branch fusion for robustness. |
| **False negatives** | Heavily post-processed authentic photos may trigger false alerts. Consider metadata context. |
| **Adversarial robustness** | Sophisticated adversaries may evade detection. Continuous model updates recommended. |
| **Cultural bias** | Detection accuracy may vary across demographics. Audit with diverse test sets. |

### Guidelines

- ❌ **Do not** use as sole evidence in legal proceedings
- ❌ **Do not** use for automated content removal without human review
- ✅ **Do** use as a screening tool in multi-step verification workflows
- ✅ **Do** combine with human expert judgment
- ✅ **Do** regularly update models and evaluation datasets

---

## 🗺️ Future Roadmap

### v1.1
- [ ] Video frame analysis support
- [ ] Batch processing API endpoint
- [ ] WebSocket real-time analysis streaming
- [ ] PDF/document forgery detection

### v1.2
- [ ] Adversarial robustness training
- [ ] Multi-GPU distributed training
- [ ] Model A/B testing framework
- [ ] Confidence calibration (Platt scaling)

### v2.0
- [ ] Blockchain-based provenance tracking (C2PA standard)
- [ ] Real-time social media monitoring integration
- [ ] Multi-tenant SaaS deployment
- [ ] Custom model fine-tuning API
- [ ] Federated learning for privacy-preserving training

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🔬 by TrueLens AI Team**

*"Trust, but verify."*

</div>
