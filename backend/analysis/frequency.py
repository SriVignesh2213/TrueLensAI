"""
TrueLens AI - Frequency Domain Analysis
Analyzes images in the frequency domain using DCT/FFT to detect AI-generated content.

AI-generated images often exhibit distinctive patterns in the frequency domain:
- GAN-generated images show periodic artifacts in the high-frequency spectrum
- Diffusion model outputs may lack certain natural frequency characteristics
- Real photographs have a specific 1/f power spectrum falloff
"""

import numpy as np
from PIL import Image
from scipy import fft as scipy_fft
import io
import base64


class FrequencyAnalyzer:
    """
    Frequency domain analyzer for AI-generated image detection.
    
    Analyzes the Discrete Fourier Transform (DFT) and Discrete Cosine
    Transform (DCT) of images to detect unnatural frequency patterns
    that are characteristic of AI-generated content.
    """
    
    def __init__(self):
        self.analysis_size = 256  # Standard analysis size
    
    def analyze(self, image: Image.Image) -> dict:
        """
        Perform comprehensive frequency domain analysis.
        
        Args:
            image: PIL Image in RGB mode
            
        Returns:
            Dictionary containing frequency analysis results
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for consistent analysis
        image_resized = image.resize((self.analysis_size, self.analysis_size))
        img_array = np.array(image_resized).astype(np.float32)
        
        # Convert to grayscale for frequency analysis
        gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # FFT Analysis
        fft_results = self._fft_analysis(gray)
        
        # Power Spectrum Analysis
        power_results = self._power_spectrum_analysis(gray)
        
        # DCT Analysis
        dct_results = self._dct_analysis(gray)
        
        # Azimuthal Average Analysis (radial power spectrum)
        azimuthal_results = self._azimuthal_analysis(gray)
        
        # Combine scores
        frequency_score = self._compute_frequency_score(
            fft_results, power_results, dct_results, azimuthal_results
        )
        
        # Generate visualization
        spectrum_b64 = self._generate_spectrum_visualization(gray)
        
        # Analysis text
        analysis_text = self._generate_analysis_text(frequency_score, fft_results, azimuthal_results)
        
        return {
            "spectrum_image_b64": spectrum_b64,
            "frequency_score": round(frequency_score, 4),
            "high_freq_energy": round(fft_results["high_freq_ratio"], 4),
            "spectral_slope": round(azimuthal_results["spectral_slope"], 4),
            "periodicity_score": round(fft_results["periodicity_score"], 4),
            "dct_energy_ratio": round(dct_results["energy_ratio"], 4),
            "analysis_text": analysis_text
        }
    
    def _fft_analysis(self, gray_image):
        """
        Perform FFT analysis to detect frequency anomalies.
        """
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log magnitude spectrum
        log_magnitude = np.log1p(magnitude)
        
        h, w = gray_image.shape
        cy, cx = h // 2, w // 2
        
        # Create frequency masks
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        
        # Low, mid, high frequency energy
        low_mask = dist < max_dist * 0.15
        mid_mask = (dist >= max_dist * 0.15) & (dist < max_dist * 0.5)
        high_mask = dist >= max_dist * 0.5
        
        total_energy = np.sum(magnitude**2)
        low_energy = np.sum(magnitude[low_mask]**2) / total_energy if total_energy > 0 else 0
        mid_energy = np.sum(magnitude[mid_mask]**2) / total_energy if total_energy > 0 else 0
        high_energy = np.sum(magnitude[high_mask]**2) / total_energy if total_energy > 0 else 0
        
        # Detect periodic artifacts (GAN fingerprints)
        periodicity_score = self._detect_periodic_artifacts(magnitude, cx, cy)
        
        return {
            "low_freq_ratio": float(low_energy),
            "mid_freq_ratio": float(mid_energy),
            "high_freq_ratio": float(high_energy),
            "periodicity_score": float(periodicity_score),
            "log_magnitude": log_magnitude
        }
    
    def _detect_periodic_artifacts(self, magnitude, cx, cy):
        """
        Detect periodic artifacts that are characteristic of GAN-generated images.
        GANs often produce periodic patterns visible as peaks in the frequency domain.
        """
        h, w = magnitude.shape
        
        # Exclude DC component and very low frequencies
        mask = np.ones_like(magnitude, dtype=bool)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask[dist < 5] = False
        
        # Get magnitude values excluding DC
        mag_filtered = magnitude.copy()
        mag_filtered[~mask] = 0
        
        # Compute statistics
        mean_mag = np.mean(mag_filtered[mask])
        std_mag = np.std(mag_filtered[mask])
        
        if std_mag == 0:
            return 0.0
        
        # Count significant peaks (values > mean + 3*std)
        threshold = mean_mag + 3 * std_mag
        peaks = np.sum(mag_filtered > threshold)
        total_pixels = np.sum(mask)
        
        # Normalize peak ratio
        peak_ratio = peaks / total_pixels if total_pixels > 0 else 0
        
        # Higher peak ratio suggests periodic artifacts
        score = min(1.0, peak_ratio * 50)  # Scale factor
        
        return score
    
    def _power_spectrum_analysis(self, gray_image):
        """
        Analyze the power spectrum for natural vs artificial patterns.
        Natural images follow a 1/f^β power law.
        """
        f_transform = np.fft.fft2(gray_image)
        power_spectrum = np.abs(f_transform) ** 2
        
        total_power = np.sum(power_spectrum)
        mean_power = np.mean(power_spectrum)
        
        return {
            "total_power": float(total_power),
            "mean_power": float(mean_power)
        }
    
    def _dct_analysis(self, gray_image):
        """
        Perform DCT analysis for compression artifact detection.
        """
        # Compute 2D DCT using scipy
        dct_coeffs = scipy_fft.dctn(gray_image, type=2, norm='ortho')
        
        h, w = gray_image.shape
        
        # Energy in different frequency bands
        total_energy = np.sum(np.abs(dct_coeffs) ** 2)
        
        # High-frequency energy (bottom-right quadrant)
        hf_energy = np.sum(np.abs(dct_coeffs[h//2:, w//2:]) ** 2)
        
        energy_ratio = hf_energy / total_energy if total_energy > 0 else 0
        
        return {
            "energy_ratio": float(energy_ratio),
            "total_energy": float(total_energy)
        }
    
    def _azimuthal_analysis(self, gray_image):
        """
        Compute radial (azimuthal) power spectrum average.
        Natural images have a characteristic 1/f^β falloff with β ≈ 2.
        AI-generated images often deviate from this pattern.
        """
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        power = np.abs(f_shift) ** 2
        
        h, w = gray_image.shape
        cy, cx = h // 2, w // 2
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        
        max_radius = min(cx, cy)
        radial_profile = np.zeros(max_radius)
        
        for r in range(1, max_radius):
            mask = dist == r
            if np.any(mask):
                radial_profile[r] = np.mean(power[mask])
        
        # Fit log-log slope (should be ~-2 for natural images)
        valid = radial_profile[1:] > 0
        if np.sum(valid) > 10:
            freqs = np.arange(1, max_radius)
            log_freqs = np.log(freqs[valid])
            log_power = np.log(radial_profile[1:][valid])
            
            # Linear fit
            coeffs = np.polyfit(log_freqs, log_power, 1)
            spectral_slope = coeffs[0]
        else:
            spectral_slope = -2.0  # Default natural value
        
        return {
            "spectral_slope": float(spectral_slope),
            "radial_profile": radial_profile.tolist()
        }
    
    def _compute_frequency_score(self, fft_results, power_results, dct_results, azimuthal_results):
        """
        Compute overall frequency-based AI detection score.
        
        Returns:
            Score from 0 (likely real) to 1 (likely AI-generated)
        """
        score = 0.0
        
        # 1. Spectral slope deviation from natural (β ≈ -2)
        slope = azimuthal_results["spectral_slope"]
        slope_deviation = abs(slope - (-2.0))
        if slope_deviation > 0.5:
            score += min(0.3, slope_deviation * 0.1)
        
        # 2. Periodic artifacts (GAN fingerprints)
        periodicity = fft_results["periodicity_score"]
        if periodicity > 0.3:
            score += periodicity * 0.3
        
        # 3. Unusual high-frequency energy
        hf_ratio = fft_results["high_freq_ratio"]
        if hf_ratio < 0.001:  # Very low HF - possible smoothing
            score += 0.2
        elif hf_ratio > 0.1:  # Very high HF - possible artifacts
            score += 0.15
        
        # 4. DCT energy distribution
        dct_ratio = dct_results["energy_ratio"]
        if dct_ratio < 0.01:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_spectrum_visualization(self, gray_image):
        """Generate frequency spectrum visualization as base64 image."""
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        
        # Normalize to 0-255
        if magnitude.max() > magnitude.min():
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
        magnitude = magnitude.astype(np.uint8)
        
        # Convert to PIL Image
        spectrum_image = Image.fromarray(magnitude, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        spectrum_image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _generate_analysis_text(self, score, fft_results, azimuthal_results):
        """Generate human-readable frequency analysis text."""
        parts = []
        
        if score > 0.6:
            parts.append("⚠️ Frequency analysis shows strong AI-generation indicators.")
        elif score > 0.3:
            parts.append("⚡ Some frequency anomalies detected that may indicate AI processing.")
        else:
            parts.append("✅ Frequency characteristics are consistent with natural photography.")
        
        slope = azimuthal_results["spectral_slope"]
        parts.append(f"Spectral slope: {slope:.2f} (natural images typically show ~-2.0).")
        
        periodicity = fft_results["periodicity_score"]
        if periodicity > 0.3:
            parts.append("Periodic artifacts detected in the frequency spectrum, possibly from GAN generation.")
        
        return " ".join(parts)
