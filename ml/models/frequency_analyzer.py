"""
TrueLens AI — Frequency Domain Forensic Analyzer

Performs spectral analysis on images using Fast Fourier Transform (FFT)
to detect frequency-domain artifacts characteristic of AI-generated content.

AI-generated images often leave spectral fingerprints:
    - GAN-generated images: periodic grid artifacts in high-frequency bands
    - Diffusion models: characteristic noise patterns in mid-frequency bands
    - Copy-move forgeries: duplicate spectral signatures

Pipeline:
    Input Image → Grayscale → 2D FFT → Log Magnitude Spectrum
    → Azimuthal Average → Feature Extraction → MLP Classifier

Author: TrueLens AI Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """
    Extracts frequency-domain features from images using 2D FFT analysis.

    Computes the magnitude spectrum, azimuthal average (radial power profile),
    and statistical features from the spectral representation.
    """

    def __init__(self, image_size: int = 224) -> None:
        """
        Initialize the spectral feature extractor.

        Args:
            image_size: Expected input image size (square).
        """
        self.image_size = image_size

    @staticmethod
    def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
        """
        Compute the log magnitude spectrum of an image via 2D FFT.

        Args:
            image: Grayscale image array of shape [H, W].

        Returns:
            Log magnitude spectrum of shape [H, W], DC-centered.
        """
        # Apply windowing to reduce spectral leakage
        rows, cols = image.shape
        row_window = np.hanning(rows)
        col_window = np.hanning(cols)
        window = np.outer(row_window, col_window)
        windowed = image * window

        # 2D FFT with DC component shifted to center
        f_transform = np.fft.fft2(windowed)
        f_shifted = np.fft.fftshift(f_transform)

        # Log magnitude spectrum (add epsilon to avoid log(0))
        magnitude = np.abs(f_shifted)
        log_magnitude = np.log1p(magnitude)

        return log_magnitude

    @staticmethod
    def azimuthal_average(spectrum: np.ndarray) -> np.ndarray:
        """
        Compute the azimuthal (radial) average of a 2D spectrum.

        This produces a 1D radial power profile, which captures how
        energy is distributed across spatial frequencies.

        Args:
            spectrum: 2D magnitude spectrum (DC-centered).

        Returns:
            1D array of radially-averaged spectral power.
        """
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2

        # Create radial distance matrix
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        max_radius = min(cy, cx)
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)

        # Accumulate values at each radius
        mask = r < max_radius
        for radius in range(max_radius):
            ring = r == radius
            if ring.any():
                radial_profile[radius] = spectrum[ring].mean()
                counts[radius] = ring.sum()

        return radial_profile

    def extract_spectral_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive spectral features from an image.

        Args:
            image: Grayscale image array [H, W] with values in [0, 255].

        Returns:
            Dictionary of spectral features:
                - 'magnitude_spectrum': 2D log magnitude spectrum
                - 'radial_profile': 1D azimuthal average
                - 'spectral_stats': Statistical features vector
                - 'high_freq_energy': High-frequency band energy ratio
        """
        # Normalize to [0, 1]
        image_norm = image.astype(np.float64) / 255.0

        # Compute FFT magnitude spectrum
        magnitude = self.compute_fft_magnitude(image_norm)

        # Radial power profile
        radial_profile = self.azimuthal_average(magnitude)

        # Statistical features
        spectral_stats = self._compute_spectral_stats(magnitude, radial_profile)

        # High-frequency energy ratio
        high_freq_energy = self._high_frequency_ratio(radial_profile)

        return {
            'magnitude_spectrum': magnitude,
            'radial_profile': radial_profile,
            'spectral_stats': spectral_stats,
            'high_freq_energy': high_freq_energy
        }

    @staticmethod
    def _compute_spectral_stats(
        magnitude: np.ndarray, radial_profile: np.ndarray
    ) -> np.ndarray:
        """
        Compute statistical descriptors of the spectrum.

        Returns a feature vector containing:
            - Mean, std, skewness, kurtosis of magnitude
            - Slope of radial profile (linear fit)
            - Energy in low/mid/high frequency bands
            - Spectral centroid and spread
        """
        from scipy import stats as sp_stats

        features = []

        # Global magnitude statistics
        features.extend([
            magnitude.mean(),
            magnitude.std(),
            float(sp_stats.skew(magnitude.ravel())),
            float(sp_stats.kurtosis(magnitude.ravel())),
        ])

        # Radial profile analysis
        n = len(radial_profile)
        if n > 1:
            x = np.arange(n)
            slope, intercept = np.polyfit(x, radial_profile, 1)
            features.extend([slope, intercept])
        else:
            features.extend([0.0, 0.0])

        # Band energy ratios (low / mid / high)
        third = max(n // 3, 1)
        total_energy = radial_profile.sum() + 1e-10
        features.extend([
            radial_profile[:third].sum() / total_energy,        # Low freq
            radial_profile[third:2*third].sum() / total_energy,  # Mid freq
            radial_profile[2*third:].sum() / total_energy,       # High freq
        ])

        # Spectral centroid
        freqs = np.arange(n)
        centroid = (freqs * radial_profile).sum() / (radial_profile.sum() + 1e-10)
        spread = np.sqrt(
            ((freqs - centroid) ** 2 * radial_profile).sum()
            / (radial_profile.sum() + 1e-10)
        )
        features.extend([centroid, spread])

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _high_frequency_ratio(radial_profile: np.ndarray) -> float:
        """
        Compute the ratio of high-frequency to total spectral energy.

        AI-generated images often have anomalous high-frequency content.
        """
        n = len(radial_profile)
        cutoff = n // 2
        total = radial_profile.sum() + 1e-10
        high = radial_profile[cutoff:].sum()
        return float(high / total)


class FrequencyClassifier(nn.Module):
    """
    MLP-based classifier for frequency-domain features.

    Takes spectral statistical features as input and outputs
    a probability score for AI-generated content.

    Architecture:
        Input (12 features) → FC 64 → ReLU → Dropout
        → FC 32 → ReLU → FC 2

    Note: This is a lightweight secondary classifier meant to augment
    the primary CNN detector. It operates on hand-crafted spectral features.
    """

    SPECTRAL_FEATURE_DIM = 12  # Number of statistical features

    def __init__(self, dropout_rate: float = 0.3) -> None:
        """
        Initialize the frequency-domain classifier.

        Args:
            dropout_rate: Dropout probability for regularization.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.SPECTRAL_FEATURE_DIM, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(32, 2)
        )
        self._init_weights()
        logger.info("FrequencyClassifier initialized")

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Spectral feature tensor of shape [B, SPECTRAL_FEATURE_DIM].

        Returns:
            Logits tensor of shape [B, 2].
        """
        return self.network(x)

    def predict_probability(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with probability scores.

        Args:
            x: Spectral feature tensor [B, SPECTRAL_FEATURE_DIM].

        Returns:
            Dictionary with 'prediction', 'confidence', 'ai_probability'.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, prediction = probs.max(dim=1)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'ai_probability': probs[:, 1]  # P(AI-generated)
        }
