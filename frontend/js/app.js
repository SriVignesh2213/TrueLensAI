/**
 * TrueLens AI - Frontend Application
 * Handles image upload, API communication, and results visualization.
 */

// ============================================================
// Configuration
// ============================================================
const API_BASE = window.location.origin;
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

// ============================================================
// DOM Elements
// ============================================================
const elements = {
    // Upload
    uploadZone: document.getElementById('uploadZone'),
    uploadContent: document.getElementById('uploadContent'),
    uploadPreview: document.getElementById('uploadPreview'),
    previewImage: document.getElementById('previewImage'),
    fileInput: document.getElementById('fileInput'),
    removeImage: document.getElementById('removeImage'),
    analyzeBtn: document.getElementById('analyzeBtn'),

    // Options
    optGradcam: document.getElementById('optGradcam'),
    optELA: document.getElementById('optELA'),
    optFrequency: document.getElementById('optFrequency'),
    optMetadata: document.getElementById('optMetadata'),
    optTexture: document.getElementById('optTexture'),

    // Results
    resultsPlaceholder: document.getElementById('resultsPlaceholder'),
    resultsContent: document.getElementById('resultsContent'),

    // Verdict
    scoreFill: document.getElementById('scoreFill'),
    scoreValue: document.getElementById('scoreValue'),
    verdictBadge: document.getElementById('verdictBadge'),
    verdictDetail: document.getElementById('verdictDetail'),
    processingTime: document.getElementById('processingTime'),
    analysisId: document.getElementById('analysisId'),

    // DL Results
    realProb: document.getElementById('realProb'),
    aiProb: document.getElementById('aiProb'),
    realBar: document.getElementById('realBar'),
    aiBar: document.getElementById('aiBar'),
    dlPrediction: document.getElementById('dlPrediction'),

    // Grad-CAM
    gradcamCard: document.getElementById('gradcamCard'),
    gradcamOverlay: document.getElementById('gradcamOverlay'),
    gradcamHeatmap: document.getElementById('gradcamHeatmap'),

    // ELA
    elaCard: document.getElementById('elaCard'),
    elaImage: document.getElementById('elaImage'),
    elaManipBar: document.getElementById('elaManipBar'),
    elaManipScore: document.getElementById('elaManipScore'),
    elaUniformBar: document.getElementById('elaUniformBar'),
    elaUniformScore: document.getElementById('elaUniformScore'),
    elaMeanError: document.getElementById('elaMeanError'),
    elaAnalysisText: document.getElementById('elaAnalysisText'),

    // Frequency
    freqCard: document.getElementById('freqCard'),
    freqSpectrum: document.getElementById('freqSpectrum'),
    freqScoreBar: document.getElementById('freqScoreBar'),
    freqScore: document.getElementById('freqScore'),
    freqSlope: document.getElementById('freqSlope'),
    freqPeriodicBar: document.getElementById('freqPeriodicBar'),
    freqPeriodic: document.getElementById('freqPeriodic'),
    freqHFEnergy: document.getElementById('freqHFEnergy'),
    freqAnalysisText: document.getElementById('freqAnalysisText'),

    // Metadata
    metaCard: document.getElementById('metaCard'),
    metaGrid: document.getElementById('metaGrid'),
    metaAnalysisText: document.getElementById('metaAnalysisText'),

    // Texture
    textureCard: document.getElementById('textureCard'),
    textureNoiseImage: document.getElementById('textureNoiseImage'),
    textureScoreBar: document.getElementById('textureScoreBar'),
    textureScore: document.getElementById('textureScore'),
    textureNoise: document.getElementById('textureNoise'),
    textureSmoothness: document.getElementById('textureSmoothness'),
    textureColorCorr: document.getElementById('textureColorCorr'),
    textureEdge: document.getElementById('textureEdge'),
    textureSatBar: document.getElementById('textureSatBar'),
    textureSat: document.getElementById('textureSat'),
    textureAnalysisText: document.getElementById('textureAnalysisText'),

    // Model warning
    modelWarning: document.getElementById('modelWarning'),

    // Image Info
    imageInfoGrid: document.getElementById('imageInfoGrid'),
};

// ============================================================
// State
// ============================================================
let selectedFile = null;
let isAnalyzing = false;

// ============================================================
// Event Listeners
// ============================================================

// Upload zone click
elements.uploadZone.addEventListener('click', (e) => {
    if (e.target === elements.removeImage || elements.removeImage.contains(e.target)) return;
    elements.fileInput.click();
});

// File input change
elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
elements.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadZone.classList.add('drag-over');
});

elements.uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    elements.uploadZone.classList.remove('drag-over');
});

elements.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Remove image
elements.removeImage.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImage();
});

// Analyze button
elements.analyzeBtn.addEventListener('click', () => {
    if (selectedFile && !isAnalyzing) {
        analyzeImage();
    }
});

// Smooth scroll for nav links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');
        if (href.startsWith('#')) {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        }
    });
});

// ============================================================
// File Handling
// ============================================================

function handleFile(file) {
    // Validate file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        showError('Invalid file type. Please upload a JPEG, PNG, WebP, or BMP image.');
        return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showError('File too large. Maximum size is 20MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadContent.style.display = 'none';
        elements.uploadPreview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Enable analyze button
    elements.analyzeBtn.disabled = false;
}

function clearImage() {
    selectedFile = null;
    elements.fileInput.value = '';
    elements.previewImage.src = '';
    elements.uploadContent.style.display = 'flex';
    elements.uploadPreview.style.display = 'none';
    elements.analyzeBtn.disabled = true;
}

// ============================================================
// API Communication
// ============================================================

async function analyzeImage() {
    if (!selectedFile || isAnalyzing) return;

    isAnalyzing = true;
    setLoadingState(true);

    // Build form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Build query params from options
    const params = new URLSearchParams({
        include_gradcam: elements.optGradcam.checked,
        include_ela: elements.optELA.checked,
        include_frequency: elements.optFrequency.checked,
        include_metadata: elements.optMetadata.checked,
        include_texture: elements.optTexture.checked,
    });

    try {
        const response = await fetch(`${API_BASE}/api/analyze?${params}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
    } finally {
        isAnalyzing = false;
        setLoadingState(false);
    }
}

// ============================================================
// Results Display
// ============================================================

function displayResults(result) {
    // Show results content
    elements.resultsPlaceholder.style.display = 'none';
    elements.resultsContent.style.display = 'flex';

    // Scroll to results
    elements.resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Display verdict
    displayVerdict(result);

    // Display deep learning results
    displayDLResults(result.deep_learning);

    // Display Grad-CAM
    if (result.gradcam && result.gradcam.available) {
        displayGradCAM(result.gradcam);
    } else {
        elements.gradcamCard.style.display = 'none';
    }

    // Display ELA
    if (result.ela && !result.ela.error) {
        displayELA(result.ela);
    } else {
        elements.elaCard.style.display = 'none';
    }

    // Display Frequency
    if (result.frequency && !result.frequency.error) {
        displayFrequency(result.frequency);
    } else {
        elements.freqCard.style.display = 'none';
    }

    // Display Metadata
    if (result.metadata && !result.metadata.error) {
        displayMetadata(result.metadata);
    } else {
        elements.metaCard.style.display = 'none';
    }

    // Display Texture/Noise Analysis
    if (result.texture && !result.texture.error) {
        displayTexture(result.texture);
    } else {
        elements.textureCard.style.display = 'none';
    }

    // Show model warning if not trained
    if (result.model_trained === false) {
        elements.modelWarning.style.display = 'block';
    } else {
        elements.modelWarning.style.display = 'none';
    }

    // Display Image Info
    displayImageInfo(result.image_info, result.filename);
}

function displayVerdict(result) {
    const score = result.authenticity_score;
    const verdict = result.verdict;

    // Animate score ring
    const circumference = 2 * Math.PI * 52; // radius = 52
    const offset = circumference - (score / 100) * circumference;

    // Add SVG gradient definition
    const svgNS = "http://www.w3.org/2000/svg";
    const scoreSvg = elements.scoreFill.closest('svg');

    // Check if gradient already exists
    let defs = scoreSvg.querySelector('defs');
    if (!defs) {
        defs = document.createElementNS(svgNS, 'defs');
        const gradient = document.createElementNS(svgNS, 'linearGradient');
        gradient.setAttribute('id', 'scoreGradient');
        gradient.setAttribute('x1', '0%');
        gradient.setAttribute('y1', '0%');
        gradient.setAttribute('x2', '100%');
        gradient.setAttribute('y2', '100%');

        let color1, color2;
        if (score >= 70) {
            color1 = '#22c55e'; color2 = '#4ade80';
        } else if (score >= 45) {
            color1 = '#f59e0b'; color2 = '#fbbf24';
        } else if (score >= 25) {
            color1 = '#f97316'; color2 = '#fb923c';
        } else {
            color1 = '#ef4444'; color2 = '#f87171';
        }

        const stop1 = document.createElementNS(svgNS, 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('stop-color', color1);
        gradient.appendChild(stop1);

        const stop2 = document.createElementNS(svgNS, 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('stop-color', color2);
        gradient.appendChild(stop2);

        defs.appendChild(gradient);
        scoreSvg.insertBefore(defs, scoreSvg.firstChild);
    }

    // Animate
    requestAnimationFrame(() => {
        elements.scoreFill.style.strokeDashoffset = offset;
    });

    // Animate score number
    animateNumber(elements.scoreValue, 0, score, 1500);

    // Set verdict badge
    elements.verdictBadge.textContent = verdict;
    elements.verdictBadge.className = 'verdict-badge';
    switch (verdict) {
        case 'AUTHENTIC':
            elements.verdictBadge.classList.add('authentic');
            break;
        case 'SUSPICIOUS':
            elements.verdictBadge.classList.add('suspicious');
            break;
        case 'LIKELY AI-GENERATED':
            elements.verdictBadge.classList.add('likely-ai');
            break;
        case 'AI-GENERATED':
            elements.verdictBadge.classList.add('ai-generated');
            break;
    }

    elements.verdictDetail.textContent = result.verdict_detail;
    elements.processingTime.textContent = `⏱ ${result.processing_time_seconds}s`;
    elements.analysisId.textContent = `ID: ${result.analysis_id}`;
}

function displayDLResults(dl) {
    elements.realProb.textContent = `${dl.real_probability}%`;
    elements.aiProb.textContent = `${dl.ai_probability}%`;

    // Animate bars
    requestAnimationFrame(() => {
        elements.realBar.style.width = `${dl.real_probability}%`;
        elements.aiBar.style.width = `${dl.ai_probability}%`;
    });

    const icon = dl.prediction === 'Real' ? '✅' : '⚠️';
    elements.dlPrediction.innerHTML = `
        ${icon} <strong>Model Prediction:</strong> ${dl.prediction} 
        (${dl.confidence}% confidence) — Using EfficientNet-B0 backbone
    `;
}

function displayGradCAM(gc) {
    elements.gradcamCard.style.display = 'block';
    elements.gradcamOverlay.src = `data:image/png;base64,${gc.overlay_image}`;
    elements.gradcamHeatmap.src = `data:image/png;base64,${gc.heatmap_image}`;
}

function displayELA(ela) {
    elements.elaCard.style.display = 'block';
    elements.elaImage.src = `data:image/png;base64,${ela.ela_image_b64}`;

    const manipPercent = (ela.manipulation_score * 100).toFixed(1);
    const uniformPercent = (ela.uniformity_score * 100).toFixed(1);

    elements.elaManipScore.textContent = `${manipPercent}%`;
    elements.elaUniformScore.textContent = `${uniformPercent}%`;
    elements.elaMeanError.textContent = ela.mean_error.toFixed(2);

    requestAnimationFrame(() => {
        elements.elaManipBar.style.width = `${manipPercent}%`;
        elements.elaUniformBar.style.width = `${uniformPercent}%`;
    });

    elements.elaAnalysisText.textContent = ela.analysis_text;
}

function displayFrequency(freq) {
    elements.freqCard.style.display = 'block';
    elements.freqSpectrum.src = `data:image/png;base64,${freq.spectrum_image_b64}`;

    const freqPercent = (freq.frequency_score * 100).toFixed(1);
    const periodicPercent = (freq.periodicity_score * 100).toFixed(1);

    elements.freqScore.textContent = `${freqPercent}%`;
    elements.freqSlope.textContent = freq.spectral_slope.toFixed(3);
    elements.freqPeriodic.textContent = `${periodicPercent}%`;
    elements.freqHFEnergy.textContent = freq.high_freq_energy.toFixed(6);

    requestAnimationFrame(() => {
        elements.freqScoreBar.style.width = `${freqPercent}%`;
        elements.freqPeriodicBar.style.width = `${periodicPercent}%`;
    });

    elements.freqAnalysisText.textContent = freq.analysis_text;
}

function displayMetadata(meta) {
    elements.metaCard.style.display = 'block';

    // Build metadata grid
    let gridHTML = '';

    gridHTML += createMetaItem('Has EXIF Data', meta.has_metadata ? 'Yes' : 'No',
        meta.has_metadata ? 'positive' : 'negative');
    gridHTML += createMetaItem('Total Tags', meta.metadata_count);

    if (meta.camera_info) {
        gridHTML += createMetaItem('Camera',
            meta.camera_info.has_camera_info ? meta.camera_info.camera_model : 'Not Found',
            meta.camera_info.has_camera_info ? 'positive' : 'negative');
    }

    if (meta.exposure_info) {
        gridHTML += createMetaItem('Exposure Data',
            meta.exposure_info.has_exposure_info ? `${meta.exposure_info.exposure_tags_found} tags` : 'Not Found',
            meta.exposure_info.has_exposure_info ? 'positive' : 'negative');
    }

    gridHTML += createMetaItem('GPS Data', meta.has_gps ? 'Present' : 'Not Found',
        meta.has_gps ? 'positive' : 'negative');

    if (meta.software_info) {
        gridHTML += createMetaItem('AI Software',
            meta.software_info.is_ai_tagged ? 'DETECTED!' : 'Not Detected',
            meta.software_info.is_ai_tagged ? 'negative' : 'positive');

        if (meta.software_info.software && meta.software_info.software !== 'None detected') {
            gridHTML += createMetaItem('Software', meta.software_info.software);
        }
    }

    gridHTML += createMetaItem('Meta Score',
        `${(meta.metadata_score * 100).toFixed(1)}%`);

    elements.metaGrid.innerHTML = gridHTML;
    elements.metaAnalysisText.textContent = meta.analysis_text;
}

function displayImageInfo(info, filename) {
    let gridHTML = '';

    gridHTML += createInfoItem('Filename', filename || 'Unknown');
    gridHTML += createInfoItem('Dimensions', `${info.width} × ${info.height}`);
    gridHTML += createInfoItem('Color Mode', info.mode);
    gridHTML += createInfoItem('Format', info.format);
    gridHTML += createInfoItem('Total Pixels', formatNumber(info.size_pixels));
    gridHTML += createInfoItem('EXIF Data', info.has_exif ? 'Present' : 'None');

    elements.imageInfoGrid.innerHTML = gridHTML;
}

function displayTexture(tex) {
    elements.textureCard.style.display = 'block';

    if (tex.noise_image_b64) {
        elements.textureNoiseImage.src = `data:image/png;base64,${tex.noise_image_b64}`;
    }

    const texPercent = (tex.texture_score * 100).toFixed(1);
    const satPercent = (tex.saturation_score * 100).toFixed(1);

    elements.textureScore.textContent = `${texPercent}%`;
    elements.textureNoise.textContent = tex.noise_level.toFixed(3);
    elements.textureSmoothness.textContent = tex.smoothness_index.toFixed(4);
    elements.textureColorCorr.textContent = tex.color_correlation.toFixed(4);
    elements.textureEdge.textContent = tex.edge_coherence.toFixed(4);
    elements.textureSat.textContent = `${satPercent}%`;

    requestAnimationFrame(() => {
        elements.textureScoreBar.style.width = `${texPercent}%`;
        elements.textureSatBar.style.width = `${satPercent}%`;
    });

    elements.textureAnalysisText.textContent = tex.analysis_text;
}

// ============================================================
// Helper Functions
// ============================================================

function createMetaItem(label, value, valueClass = '') {
    return `
        <div class="meta-item">
            <span class="meta-item-label">${label}</span>
            <span class="meta-item-value ${valueClass}">${value}</span>
        </div>
    `;
}

function createInfoItem(label, value) {
    return `
        <div class="info-item">
            <div class="info-item-label">${label}</div>
            <div class="info-item-value">${value}</div>
        </div>
    `;
}

function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + range * eased);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function setLoadingState(loading) {
    const btnContent = elements.analyzeBtn.querySelector('.btn-content');
    const btnLoading = elements.analyzeBtn.querySelector('.btn-loading');

    if (loading) {
        btnContent.style.display = 'none';
        btnLoading.style.display = 'flex';
        elements.analyzeBtn.disabled = true;
    } else {
        btnContent.style.display = 'flex';
        btnLoading.style.display = 'none';
        elements.analyzeBtn.disabled = !selectedFile;
    }
}

function showError(message) {
    // Remove existing toasts
    document.querySelectorAll('.error-toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// ============================================================
// Intersection Observer for animations
// ============================================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animation = 'fadeInUp 0.6s ease forwards';
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe feature cards, pipeline steps, usecase cards
document.querySelectorAll('.feature-card, .pipeline-step, .usecase-card').forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
});

// ============================================================
// Active nav link on scroll
// ============================================================

const sections = document.querySelectorAll('section[id]');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        if (scrollY >= sectionTop) {
            current = section.getAttribute('id');
        }
    });

    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// ============================================================
// Health Check on Load
// ============================================================

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            console.log('✅ TrueLens AI backend is healthy');
            console.log(`   Device: ${data.device}`);
            console.log(`   CUDA: ${data.cuda_available}`);
        }
    } catch (error) {
        console.warn('⚠️ Backend not reachable. Make sure the FastAPI server is running.');
    }
}

// Run health check when page loads
window.addEventListener('load', checkHealth);
