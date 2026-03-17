"""
TrueLens AI - Texture & Noise Pattern Analysis
Detects AI-generated images by analyzing noise distribution, texture smoothness,
color statistics, and edge coherence.

AI-generated images (especially from diffusion models like Gemini, DALL-E, 
Stable Diffusion) have specific statistical fingerprints:
- Unnaturally smooth noise patterns
- Different color channel correlations
- Unusual edge coherence (too perfect or too noisy)
- Abnormal local variance distributions
"""

import numpy as np
from PIL import Image, ImageFilter
import io
import base64


class TextureAnalyzer:
    """
    Analyzes image texture and noise patterns to detect AI generation.
    
    AI-generated images differ from real photographs in several ways:
    1. Noise patterns: Real cameras produce sensor noise; AI images lack this
    2. Texture smoothness: AI images often have unnatural micro-texture
    3. Color statistics: AI images have different channel correlations
    4. Edge coherence: AI edges can be too smooth or show artifacts
    5. Local variance: Real images have natural variance distributions
    """
    
    def __init__(self):
        self.analysis_size = 512  # Higher res for texture analysis
    
    def analyze(self, image: Image.Image) -> dict:
        """
        Perform comprehensive texture and noise analysis.
        
        Args:
            image: PIL Image in RGB mode
            
        Returns:
            Dictionary with texture analysis results
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for consistent analysis
        img_resized = image.resize(
            (self.analysis_size, self.analysis_size),
            Image.LANCZOS
        )
        img_array = np.array(img_resized).astype(np.float64)
        
        # 1. Noise pattern analysis
        noise_result = self._analyze_noise_patterns(img_array)
        
        # 2. Texture smoothness
        smoothness_result = self._analyze_smoothness(img_array, img_resized)
        
        # 3. Color statistics
        color_result = self._analyze_color_statistics(img_array)
        
        # 4. Edge coherence
        edge_result = self._analyze_edge_coherence(img_resized)
        
        # 5. Local variance analysis
        variance_result = self._analyze_local_variance(img_array)
        
        # 6. Saturation analysis
        saturation_result = self._analyze_saturation(img_array)
        
        # Compute overall texture score
        texture_score = self._compute_texture_score(
            noise_result, smoothness_result, color_result,
            edge_result, variance_result, saturation_result
        )
        
        # Generate noise visualization
        noise_vis_b64 = self._generate_noise_visualization(img_array)
        
        # Analysis text
        analysis_text = self._generate_analysis_text(
            texture_score, noise_result, smoothness_result,
            color_result, saturation_result
        )
        
        return {
            "texture_score": round(texture_score, 4),
            "noise_level": round(noise_result["noise_level"], 4),
            "noise_uniformity": round(noise_result["noise_uniformity"], 4),
            "smoothness_index": round(smoothness_result["smoothness_index"], 4),
            "color_correlation": round(color_result["channel_correlation"], 4),
            "edge_coherence": round(edge_result["coherence_score"], 4),
            "variance_kurtosis": round(variance_result["kurtosis"], 4),
            "saturation_score": round(saturation_result["saturation_ai_score"], 4),
            "noise_image_b64": noise_vis_b64,
            "analysis_text": analysis_text
        }
    
    def _analyze_noise_patterns(self, img_array):
        """
        Analyze noise patterns in the image.
        Real camera images have characteristic sensor noise;
        AI images have unnaturally smooth or synthetic noise.
        """
        # Extract noise by subtracting the smoothed version
        # Use a simple 5x5 median-like approach via convolution
        from scipy.ndimage import uniform_filter, median_filter
        
        gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # Extract noise residual
        smoothed = uniform_filter(gray, size=5)
        noise = gray - smoothed
        
        # Noise statistics
        noise_level = float(np.std(noise))
        noise_mean = float(np.mean(np.abs(noise)))
        
        # Noise uniformity - divide image into blocks and check noise consistency
        h, w = noise.shape
        block_size = 32
        block_stds = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise[y:y+block_size, x:x+block_size]
                block_stds.append(float(np.std(block)))
        
        if block_stds:
            noise_uniformity = 1.0 - min(1.0, float(np.std(block_stds)) / (float(np.mean(block_stds)) + 1e-8))
        else:
            noise_uniformity = 0.5
        
        # Check for noise-free regions (AI images often have very clean areas)
        clean_blocks = sum(1 for s in block_stds if s < 1.0)
        clean_ratio = clean_blocks / (len(block_stds) + 1e-8)
        
        # Noise histogram entropy
        noise_clipped = np.clip(noise, -30, 30)
        hist, _ = np.histogram(noise_clipped, bins=64)
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        return {
            "noise_level": noise_level,
            "noise_mean": noise_mean,
            "noise_uniformity": noise_uniformity,
            "clean_ratio": clean_ratio,
            "noise_entropy": float(entropy),
            "noise_residual": noise
        }
    
    def _analyze_smoothness(self, img_array, img_pil):
        """
        Analyze texture smoothness.
        AI images tend to be unnaturally smooth in certain regions
        or have a specific kind of synthetic texture.
        """
        gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # Laplacian variance (measure of overall sharpness/texture)
        from scipy.ndimage import laplace
        laplacian = laplace(gray)
        laplacian_var = float(np.var(laplacian))
        
        # Local gradient magnitude
        gy = np.diff(gray, axis=0)
        gx = np.diff(gray, axis=1)
        min_dim = min(gy.shape[0], gx.shape[0])
        min_dim_w = min(gy.shape[1], gx.shape[1])
        gradient_mag = np.sqrt(gy[:min_dim, :min_dim_w]**2 + gx[:min_dim, :min_dim_w]**2)
        
        avg_gradient = float(np.mean(gradient_mag))
        gradient_std = float(np.std(gradient_mag))
        
        # Smoothness index: ratio of very smooth patches
        h, w = gray.shape
        block_size = 16
        smooth_blocks = 0
        total_blocks = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                if np.std(block) < 3.0:  # Very smooth
                    smooth_blocks += 1
                total_blocks += 1
        
        smoothness_index = smooth_blocks / (total_blocks + 1e-8)
        
        return {
            "laplacian_variance": laplacian_var,
            "avg_gradient": avg_gradient,
            "gradient_std": gradient_std,
            "smoothness_index": smoothness_index
        }
    
    def _analyze_color_statistics(self, img_array):
        """
        Analyze color channel statistics and correlations.
        AI images often have unusual color distributions.
        """
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Channel means and stds
        means = [float(np.mean(c)) for c in [r, g, b]]
        stds = [float(np.std(c)) for c in [r, g, b]]
        
        # Inter-channel correlation
        r_flat, g_flat, b_flat = r.flatten(), g.flatten(), b.flatten()
        rg_corr = float(np.corrcoef(r_flat, g_flat)[0, 1])
        rb_corr = float(np.corrcoef(r_flat, b_flat)[0, 1])
        gb_corr = float(np.corrcoef(g_flat, b_flat)[0, 1])
        
        avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
        
        # Color histogram smoothness
        hist_smoothness = []
        for channel in [r, g, b]:
            hist, _ = np.histogram(channel, bins=256, range=(0, 255))
            hist_diff = np.diff(hist.astype(np.float64))
            hist_smoothness.append(float(np.std(hist_diff)))
        
        avg_hist_roughness = float(np.mean(hist_smoothness))
        
        # Unique color ratio
        combined = img_array.reshape(-1, 3)
        # Sample for speed
        if len(combined) > 50000:
            indices = np.random.choice(len(combined), 50000, replace=False)
            combined_sample = combined[indices]
        else:
            combined_sample = combined
        
        # Quantize to reduce exact matches
        quantized = (combined_sample / 4).astype(np.int32)
        unique_colors = len(np.unique(quantized.view(np.dtype((np.void, quantized.dtype.itemsize * 3)))))
        color_diversity = unique_colors / len(quantized)
        
        return {
            "channel_means": means,
            "channel_stds": stds,
            "channel_correlation": avg_corr,
            "rg_correlation": rg_corr,
            "rb_correlation": rb_corr,
            "gb_correlation": gb_corr,
            "hist_roughness": avg_hist_roughness,
            "color_diversity": float(color_diversity)
        }
    
    def _analyze_edge_coherence(self, img_pil):
        """
        Analyze edge patterns for AI generation artifacts.
        """
        # Apply edge detection
        edges = img_pil.convert('L').filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges).astype(np.float64)
        
        # Edge density
        edge_threshold = 30
        edge_mask = edge_array > edge_threshold
        edge_density = float(np.mean(edge_mask))
        
        # Edge continuity (connected edge analysis)
        # Check how many edge pixels have neighboring edge pixels
        if np.any(edge_mask):
            edge_coords = np.argwhere(edge_mask)
            
            # Sample edges for speed
            if len(edge_coords) > 5000:
                indices = np.random.choice(len(edge_coords), 5000, replace=False)
                edge_sample = edge_coords[indices]
            else:
                edge_sample = edge_coords
            
            connected = 0
            h, w = edge_mask.shape
            for y, x in edge_sample:
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and edge_mask[ny, nx]:
                            neighbors += 1
                if neighbors >= 2:
                    connected += 1
            
            coherence = connected / len(edge_sample) if len(edge_sample) > 0 else 0.5
        else:
            coherence = 0.0
        
        # Edge gradation (smooth vs sharp transitions)
        edge_mean = float(np.mean(edge_array))
        edge_std = float(np.std(edge_array))
        
        return {
            "edge_density": edge_density,
            "coherence_score": float(coherence),
            "edge_mean": edge_mean,
            "edge_std": edge_std
        }
    
    def _analyze_local_variance(self, img_array):
        """
        Analyze local variance distribution.
        Real images have a specific distribution of local variances;
        AI images may deviate from this natural distribution.
        """
        from scipy.ndimage import uniform_filter
        
        gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # Compute local variance using a sliding window
        win_size = 7
        local_mean = uniform_filter(gray, size=win_size)
        local_sqmean = uniform_filter(gray**2, size=win_size)
        local_var = local_sqmean - local_mean**2
        local_var = np.maximum(local_var, 0)  # Ensure non-negative
        
        # Statistics of local variance
        mean_var = float(np.mean(local_var))
        std_var = float(np.std(local_var))
        
        # Kurtosis of local variance distribution
        # Higher kurtosis = more peaked = more uniform = more AI-like
        if std_var > 0:
            centered = local_var.flatten() - mean_var
            kurtosis = float(np.mean(centered**4) / (std_var**4) - 3)
        else:
            kurtosis = 0.0
        
        # Skewness
        if std_var > 0:
            skewness = float(np.mean(centered**3) / (std_var**3))
        else:
            skewness = 0.0
        
        # Ratio of very low variance pixels (artificially smooth areas)
        low_var_ratio = float(np.mean(local_var < 5.0))
        
        return {
            "mean_variance": mean_var,
            "variance_std": std_var,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "low_variance_ratio": low_var_ratio
        }
    
    def _analyze_saturation(self, img_array):
        """
        Analyze color saturation patterns.
        AI images often have unnaturally vibrant or uniform saturation.
        """
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Compute saturation (simplified HSV saturation)
        max_ch = np.maximum(np.maximum(r, g), b)
        min_ch = np.minimum(np.minimum(r, g), b)
        
        # Avoid division by zero
        denominator = max_ch.copy()
        denominator[denominator == 0] = 1
        
        saturation = (max_ch - min_ch) / denominator
        
        mean_sat = float(np.mean(saturation))
        std_sat = float(np.std(saturation))
        
        # High saturation ratio
        high_sat_ratio = float(np.mean(saturation > 0.7))
        
        # Saturation uniformity (AI tends to be more uniform)
        h, w = saturation.shape
        block_size = 32
        sat_stds = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = saturation[y:y+block_size, x:x+block_size]
                sat_stds.append(float(np.std(block)))
        
        sat_uniformity = 1.0 - min(1.0, float(np.std(sat_stds)) / (float(np.mean(sat_stds)) + 1e-8)) if sat_stds else 0.5
        
        # AI score based on saturation patterns
        ai_score = 0.0
        if mean_sat > 0.45:  # Overly saturated
            ai_score += 0.3
        if high_sat_ratio > 0.25:
            ai_score += 0.2
        if sat_uniformity > 0.75:  # Too uniform saturation
            ai_score += 0.2
        if std_sat < 0.15:  # Very narrow saturation range
            ai_score += 0.15
        
        return {
            "mean_saturation": mean_sat,
            "saturation_std": std_sat,
            "high_saturation_ratio": high_sat_ratio,
            "saturation_uniformity": sat_uniformity,
            "saturation_ai_score": min(1.0, ai_score)
        }
    
    def _compute_texture_score(self, noise_r, smooth_r, color_r, edge_r, var_r, sat_r):
        """
        Compute overall texture-based AI detection score.
        
        Returns 0 (likely real) to 1 (likely AI-generated)
        """
        score = 0.0
        indicators = 0
        total_weight = 0
        
        # 1. Noise analysis (weight: 0.25)
        w = 0.25
        noise_score = 0.0
        
        # Very low noise = likely AI
        if noise_r["noise_level"] < 2.5:
            noise_score += 0.5
        elif noise_r["noise_level"] < 4.0:
            noise_score += 0.3
        
        # Very uniform noise = likely AI  
        if noise_r["noise_uniformity"] > 0.8:
            noise_score += 0.3
        
        # Many clean blocks = likely AI
        if noise_r["clean_ratio"] > 0.3:
            noise_score += 0.2
        
        score += min(1.0, noise_score) * w
        total_weight += w
        
        # 2. Smoothness (weight: 0.2)
        w = 0.2
        smooth_score = 0.0
        
        if smooth_r["smoothness_index"] > 0.15:
            smooth_score += 0.4
        if smooth_r["laplacian_variance"] < 50:
            smooth_score += 0.3
        elif smooth_r["laplacian_variance"] < 200:
            smooth_score += 0.15
        if smooth_r["avg_gradient"] < 8.0:
            smooth_score += 0.3
        
        score += min(1.0, smooth_score) * w
        total_weight += w
        
        # 3. Color statistics (weight: 0.2)
        w = 0.2
        color_score = 0.0
        
        # Very high channel correlation common in AI
        if color_r["channel_correlation"] > 0.92:
            color_score += 0.3
        
        # Smooth color histogram = AI
        if color_r["hist_roughness"] < 100:
            color_score += 0.3
        
        # Low color diversity = AI
        if color_r["color_diversity"] < 0.4:
            color_score += 0.2
        
        score += min(1.0, color_score) * w
        total_weight += w
        
        # 4. Edge coherence (weight: 0.1)
        w = 0.1
        edge_score = 0.0
        
        if edge_r["coherence_score"] > 0.85:
            edge_score += 0.4  # Unnaturally coherent edges
        if edge_r["edge_density"] < 0.05:
            edge_score += 0.3  # Too few edges
        
        score += min(1.0, edge_score) * w
        total_weight += w
        
        # 5. Local variance (weight: 0.15)
        w = 0.15
        var_score = 0.0
        
        if var_r["kurtosis"] > 10:
            var_score += 0.3  # Too peaked distribution
        if var_r["low_variance_ratio"] > 0.4:
            var_score += 0.4  # Too many smooth areas
        if var_r["skewness"] > 3:
            var_score += 0.2
        
        score += min(1.0, var_score) * w
        total_weight += w
        
        # 6. Saturation (weight: 0.1)
        w = 0.1
        score += min(1.0, sat_r["saturation_ai_score"]) * w
        total_weight += w
        
        return min(1.0, score / total_weight) if total_weight > 0 else 0.5
    
    def _generate_noise_visualization(self, img_array):
        """Generate a noise residual visualization."""
        from scipy.ndimage import uniform_filter
        
        gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        smoothed = uniform_filter(gray, size=3)
        noise = gray - smoothed
        
        # Scale noise for visualization
        noise_scaled = np.clip((noise + 15) * (255.0 / 30.0), 0, 255).astype(np.uint8)
        
        noise_img = Image.fromarray(noise_scaled, mode='L')
        # Resize for display
        noise_img = noise_img.resize((256, 256), Image.NEAREST)
        
        buffer = io.BytesIO()
        noise_img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _generate_analysis_text(self, score, noise_r, smooth_r, color_r, sat_r):
        """Generate human-readable analysis text."""
        parts = []
        
        if score > 0.6:
            parts.append("⚠️ Texture analysis shows strong AI-generation indicators.")
        elif score > 0.35:
            parts.append("⚡ Some texture anomalies detected that may indicate AI processing.")
        else:
            parts.append("✅ Texture patterns are consistent with natural photography.")
        
        if noise_r["noise_level"] < 3.0:
            parts.append("Very low sensor noise detected — real camera photos typically exhibit natural noise.")
        
        if noise_r["noise_uniformity"] > 0.8:
            parts.append("Noise distribution is unusually uniform across the image.")
        
        if smooth_r["smoothness_index"] > 0.15:
            parts.append("Image contains abnormally smooth regions typical of AI generation.")
        
        if sat_r["saturation_ai_score"] > 0.4:
            parts.append("Saturation patterns appear artificially enhanced or uniform.")
        
        if color_r["channel_correlation"] > 0.92:
            parts.append("Color channels show unnaturally high correlation.")
        
        return " ".join(parts)
