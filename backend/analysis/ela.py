"""
TrueLens AI - Error Level Analysis (ELA)
Detects image manipulations by analyzing JPEG compression artifacts.

ELA works by re-saving an image at a known quality level and then computing
the difference between the original and re-saved versions. Manipulated regions
will show different error levels compared to the rest of the image.
"""

import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import base64


class ELAAnalyzer:
    """
    Error Level Analysis for detecting image manipulations.
    
    When a JPEG image is saved, the entire image is compressed at the same
    quality level. If parts of the image have been modified (e.g., copy-paste,
    AI inpainting), those regions will have different error levels when the
    image is re-compressed.
    
    AI-generated images tend to have very uniform error levels since they
    were never JPEG-compressed in their creation pipeline.
    """
    
    def __init__(self, quality=90, scale=15):
        """
        Initialize ELA analyzer.
        
        Args:
            quality: JPEG quality level for re-compression (1-100)
            scale: Scale factor for enhancing the difference image
        """
        self.quality = quality
        self.scale = scale
    
    def analyze(self, image: Image.Image) -> dict:
        """
        Perform Error Level Analysis on an image.
        Uses multi-quality ELA for more robust detection.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Primary ELA at configured quality
        ela_image, error_magnitude = self._compute_ela(image, self.quality)
        
        mean_error = float(np.mean(error_magnitude))
        std_error = float(np.std(error_magnitude))
        max_error = float(np.max(error_magnitude))
        
        # Uniformity score: low std relative to mean indicates uniform errors
        if mean_error > 0:
            uniformity_score = 1.0 - min(1.0, std_error / (mean_error + 1e-8))
        else:
            uniformity_score = 1.0
        
        # Multi-quality ELA for robustness
        multi_quality_scores = self._multi_quality_ela(image)
        
        # Block artifact analysis
        block_score = self._analyze_block_artifacts(error_magnitude)
        
        # Spatial coherence analysis
        spatial_score = self._analyze_spatial_coherence(error_magnitude)
        
        # Error entropy analysis
        entropy_score = self._analyze_error_entropy(error_magnitude)
        
        # Comprehensive manipulation score
        manipulation_score = self._compute_manipulation_score(
            error_magnitude, mean_error, std_error, max_error,
            uniformity_score, block_score, spatial_score,
            entropy_score, multi_quality_scores
        )
        
        # Scale ELA image for visualization
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        
        scale = 255.0 / max_diff
        ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale * 0.5)
        
        # Convert to base64
        ela_b64 = self._image_to_base64(ela_enhanced)
        
        # Generate analysis text
        analysis_text = self._generate_analysis_text(
            mean_error, std_error, uniformity_score, manipulation_score,
            block_score, spatial_score
        )
        
        return {
            "ela_image_b64": ela_b64,
            "mean_error": round(mean_error, 4),
            "std_error": round(std_error, 4),
            "max_error": round(max_error, 4),
            "uniformity_score": round(uniformity_score, 4),
            "manipulation_score": round(manipulation_score, 4),
            "analysis_text": analysis_text
        }
    
    def _compute_ela(self, image, quality):
        """Compute ELA at a specific quality level."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        ela_image = ImageChops.difference(image, resaved)
        ela_array = np.array(ela_image).astype(np.float32)
        error_magnitude = np.sqrt(np.sum(ela_array ** 2, axis=2))
        
        return ela_image, error_magnitude
    
    def _multi_quality_ela(self, image):
        """
        Perform ELA at multiple quality levels.
        AI-generated images show consistent behavior across qualities,
        while real JPEG images show quality-dependent patterns.
        """
        qualities = [70, 80, 90, 95]
        mean_errors = []
        
        for q in qualities:
            _, error_mag = self._compute_ela(image, q)
            mean_errors.append(float(np.mean(error_mag)))
        
        # Check if error scales consistently (real images do this more)
        if len(mean_errors) >= 2 and mean_errors[-1] > 0:
            # Ratio of error at low quality vs high quality
            quality_ratio = mean_errors[0] / (mean_errors[-1] + 1e-8)
            
            # Real JPEG images show much higher error at lower quality
            # AI images (PNG/no JPEG history) show more uniform errors across qualities
            error_consistency = float(np.std(mean_errors) / (np.mean(mean_errors) + 1e-8))
        else:
            quality_ratio = 1.0
            error_consistency = 0.5
        
        return {
            "mean_errors": mean_errors,
            "quality_ratio": quality_ratio,
            "error_consistency": error_consistency
        }
    
    def _analyze_spatial_coherence(self, error_magnitude):
        """
        Analyze spatial coherence of error levels.
        AI images have very spatially coherent (smooth) error distributions.
        Real photos with JPEG history have more structured block patterns.
        """
        h, w = error_magnitude.shape
        block_size = 16
        block_means = []
        
        for y in range(0, h - block_size, block_size):
            row_means = []
            for x in range(0, w - block_size, block_size):
                block = error_magnitude[y:y+block_size, x:x+block_size]
                row_means.append(float(np.mean(block)))
            block_means.append(row_means)
        
        if not block_means or not block_means[0]:
            return 0.5
        
        block_array = np.array(block_means)
        
        # Compute variation between neighboring blocks
        neighbor_diffs = []
        rows, cols = block_array.shape
        for y in range(rows):
            for x in range(cols):
                if x < cols - 1:
                    neighbor_diffs.append(abs(block_array[y, x] - block_array[y, x+1]))
                if y < rows - 1:
                    neighbor_diffs.append(abs(block_array[y, x] - block_array[y+1, x]))
        
        if not neighbor_diffs:
            return 0.5
        
        mean_diff = float(np.mean(neighbor_diffs))
        overall_mean = float(np.mean(block_array))
        
        # Very low neighbor differences relative to overall mean = very coherent = AI
        if overall_mean > 0:
            coherence = 1.0 - min(1.0, mean_diff / (overall_mean + 1e-8))
        else:
            coherence = 1.0
        
        return coherence
    
    def _analyze_error_entropy(self, error_magnitude):
        """
        Analyze the entropy of the error distribution.
        AI images tend to have lower entropy (more predictable errors).
        """
        # Quantize error values
        max_val = max(float(np.max(error_magnitude)), 1.0)
        quantized = (error_magnitude / max_val * 63).astype(np.int32)
        quantized = np.clip(quantized, 0, 63)
        
        # Compute histogram
        hist = np.bincount(quantized.flatten(), minlength=64).astype(np.float64)
        hist = hist / (hist.sum() + 1e-8)
        
        # Shannon entropy
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        max_entropy = np.log2(64)  # Maximum possible entropy
        
        # Normalized entropy (0 = perfectly uniform/concentrated, 1 = maximally random)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _compute_manipulation_score(self, error_mag, mean_err, std_err, max_err,
                                     uniformity, block_score, spatial_score,
                                     entropy_score, multi_q):
        """
        Compute manipulation/AI-generation likelihood score.
        Uses multiple indicators with proper weighting for modern AI generators.
        """
        score = 0.0
        
        # 1. Error uniformity (weight: high for AI detection)
        #    AI-generated images have very uniform error levels
        if uniformity > 0.7:
            score += 0.15
        if uniformity > 0.5:
            score += 0.1
        
        # 2. Very low mean error = never been JPEG compressed = likely AI
        if mean_err < 1.5:
            score += 0.15
        elif mean_err < 3.0:
            score += 0.1
        elif mean_err < 5.0:
            score += 0.05
        
        # 3. Block artifact analysis
        #    Score > 0.5 means lacking natural JPEG structure
        if block_score > 0.6:
            score += 0.15
        elif block_score > 0.5:
            score += 0.08
        
        # 4. Spatial coherence
        #    Very high coherence = AI-like smooth error distribution
        if spatial_score > 0.8:
            score += 0.15
        elif spatial_score > 0.6:
            score += 0.08
        
        # 5. Error entropy
        #    Very low entropy = concentrated/predictable errors = AI
        if entropy_score < 0.3:
            score += 0.1
        elif entropy_score < 0.5:
            score += 0.05
        
        # 6. Multi-quality consistency
        #    Low consistency = errors don't change much with quality = no JPEG history
        if multi_q["error_consistency"] < 0.3:
            score += 0.1
        
        # Low quality ratio means similar error at all qualities = likely AI
        if multi_q["quality_ratio"] < 2.0:
            score += 0.1
        
        # 7. Max error relative to mean (large spikes suggest splicing)
        if max_err > mean_err * 8 and mean_err > 1.0:
            score += 0.1
        
        # 8. High standard deviation relative to mean (mixed regions)
        if mean_err > 0 and std_err / mean_err > 2.0:
            score += 0.08
        
        return min(1.0, score)
    
    def _analyze_block_artifacts(self, error_magnitude):
        """
        Analyze JPEG block artifacts in the error image.
        AI-generated images typically lack natural JPEG 8x8 block boundaries.
        """
        h, w = error_magnitude.shape
        
        if h < 24 or w < 24:
            return 0.5
        
        # Analyze in larger region for better statistics
        analysis_h = min(h, 128)
        analysis_w = min(w, 128)
        
        boundary_diffs = []
        interior_diffs = []
        
        for y in range(1, analysis_h):
            for x in range(1, analysis_w):
                diff_v = abs(float(error_magnitude[y, x]) - float(error_magnitude[y-1, x]))
                diff_h = abs(float(error_magnitude[y, x]) - float(error_magnitude[y, x-1]))
                avg_diff = (diff_v + diff_h) / 2
                
                if y % 8 == 0 or x % 8 == 0:
                    boundary_diffs.append(avg_diff)
                else:
                    interior_diffs.append(avg_diff)
        
        if not boundary_diffs or not interior_diffs:
            return 0.5
        
        avg_boundary = float(np.mean(boundary_diffs))
        avg_interior = float(np.mean(interior_diffs))
        
        # If boundary and interior diffs are similar, lacks JPEG structure  
        if avg_interior > 0:
            ratio = avg_boundary / avg_interior
            if ratio < 1.1:     # No block structure at all
                return 0.8      # Very likely AI-generated
            elif ratio < 1.3:   # Weak block structure
                return 0.65     # Possibly AI
            elif ratio > 2.0:   # Strong block structure
                return 0.2      # Likely real JPEG
            elif ratio > 1.5:
                return 0.35     # Probably real
        
        return 0.5
    
    def _generate_analysis_text(self, mean_error, std_error, uniformity,
                                 manipulation, block_score, spatial_score):
        """Generate human-readable analysis text."""
        parts = []
        
        if manipulation > 0.6:
            parts.append("⚠️ HIGH RISK: Strong indicators of AI generation or manipulation detected.")
        elif manipulation > 0.35:
            parts.append("⚡ MODERATE RISK: Some indicators of possible manipulation found.")
        else:
            parts.append("✅ LOW RISK: Error levels appear consistent with a natural photograph.")
        
        if uniformity > 0.6:
            parts.append("Error levels are unusually uniform, characteristic of AI-generated images.")
        elif uniformity < 0.25:
            parts.append("Error levels show high variation, which could indicate splicing or editing.")
        
        if mean_error < 3.0:
            parts.append("Very low compression artifacts suggest the image may not have a natural JPEG history.")
        
        if block_score > 0.6:
            parts.append("Image lacks natural JPEG 8×8 block compression patterns.")
        
        if spatial_score > 0.7:
            parts.append("Error distribution is abnormally smooth across the image.")
        
        return " ".join(parts)
    
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
