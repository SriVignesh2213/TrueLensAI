"""
TrueLens AI — Forgery Localization Engine

Performs pixel-level forgery detection using multiple techniques:
    1. Error Level Analysis (ELA) — detects inconsistent JPEG compression
    2. Grad-CAM heatmaps — highlights CNN attention regions
    3. Region merging — combines evidence into bounding boxes

Output:
    - Pixel-level heatmap (float32, [0, 1])
    - Bounding boxes around suspicious regions
    - Binary mask of forged regions
    - Overlay visualization

Author: TrueLens AI Team
License: MIT
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image, ImageFilter
import io
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForgeryRegion:
    """
    Represents a detected suspicious region.

    Attributes:
        bbox: Bounding box (x_min, y_min, x_max, y_max) in pixel coordinates.
        confidence: Confidence score for this region [0, 1].
        area_ratio: Ratio of region area to total image area.
        label: Description of the detected anomaly type.
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    area_ratio: float
    label: str = "Suspicious Region"

    def to_dict(self) -> Dict:
        return {
            'bbox': list(self.bbox),
            'confidence': round(self.confidence, 4),
            'area_ratio': round(self.area_ratio, 4),
            'label': self.label,
        }


@dataclass
class ForgeryLocalizationResult:
    """
    Complete forgery localization analysis result.

    Attributes:
        heatmap: Pixel-level suspicion heatmap [H, W], values in [0, 1].
        regions: List of detected suspicious regions.
        mask: Binary mask of suspicious areas [H, W], bool.
        manipulation_score: Overall manipulation likelihood [0, 1].
        ela_map: Error Level Analysis map (raw).
    """
    heatmap: np.ndarray
    regions: List[ForgeryRegion] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    manipulation_score: float = 0.0
    ela_map: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            'manipulation_score': round(self.manipulation_score, 4),
            'num_suspicious_regions': len(self.regions),
            'regions': [r.to_dict() for r in self.regions],
            'heatmap_shape': list(self.heatmap.shape),
        }


class ErrorLevelAnalyzer:
    """
    Error Level Analysis (ELA) for JPEG forgery detection.

    ELA works by re-saving the image at a known quality level and
    computing the difference with the original. Authentic images show
    uniform error levels, while manipulated regions show significantly
    different error levels.
    """

    def __init__(self, quality: int = 90, amplification: int = 15) -> None:
        """
        Initialize ELA analyzer.

        Args:
            quality: JPEG re-compression quality (1-100).
            amplification: Error amplification factor for visualization.
        """
        self.quality = quality
        self.amplification = amplification

    def compute_ela(self, image_path: str) -> np.ndarray:
        """
        Compute Error Level Analysis map for an image.

        Args:
            image_path: Path to the input image.

        Returns:
            ELA difference map as float32 array [H, W, 3], values in [0, 1].
        """
        original = Image.open(image_path).convert('RGB')

        # Re-compress at specified quality
        buffer = io.BytesIO()
        original.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert('RGB')

        # Compute absolute difference
        orig_array = np.array(original, dtype=np.float32)
        recomp_array = np.array(recompressed, dtype=np.float32)

        ela = np.abs(orig_array - recomp_array)

        # Amplify for visualization
        ela_amplified = ela * self.amplification

        # Normalize to [0, 1]
        ela_amplified = np.clip(ela_amplified / 255.0, 0.0, 1.0)

        return ela_amplified

    def compute_ela_grayscale(self, image_path: str) -> np.ndarray:
        """
        Compute grayscale ELA map (mean across channels).

        Args:
            image_path: Path to the input image.

        Returns:
            Grayscale ELA map [H, W], values in [0, 1].
        """
        ela_rgb = self.compute_ela(image_path)
        return ela_rgb.mean(axis=2)


class ForgeryLocalizer:
    """
    Multi-technique forgery localization engine.

    Combines Error Level Analysis, Grad-CAM heatmaps (if available),
    and statistical analysis to produce comprehensive forgery localization.

    The final heatmap is a weighted fusion of:
        - ELA map (weight: 0.4)
        - Grad-CAM heatmap (weight: 0.4, if available)
        - Statistical deviation map (weight: 0.2)
    """

    def __init__(
        self,
        ela_quality: int = 90,
        ela_amplification: int = 15,
        threshold: float = 0.5,
        min_region_area: int = 500
    ) -> None:
        """
        Initialize the forgery localizer.

        Args:
            ela_quality: JPEG quality for ELA re-compression.
            ela_amplification: ELA amplification factor.
            threshold: Binary threshold for suspicious region detection.
            min_region_area: Minimum pixel area for a valid suspicious region.
        """
        self.ela = ErrorLevelAnalyzer(quality=ela_quality, amplification=ela_amplification)
        self.threshold = threshold
        self.min_region_area = min_region_area
        logger.info("ForgeryLocalizer initialized")

    def localize(
        self,
        image_path: str,
        gradcam_heatmap: Optional[np.ndarray] = None
    ) -> ForgeryLocalizationResult:
        """
        Perform full forgery localization analysis.

        Args:
            image_path: Path to the input image.
            gradcam_heatmap: Optional Grad-CAM heatmap [H, W] from CNN detector.

        Returns:
            ForgeryLocalizationResult with heatmap, regions, and mask.
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        h, w = img_array.shape[:2]

        # === Compute ELA Map ===
        ela_map = self.ela.compute_ela_grayscale(image_path)

        # === Compute Statistical Deviation Map ===
        stat_map = self._compute_statistical_map(img_array)

        # === Fuse Heatmaps ===
        if gradcam_heatmap is not None:
            # Resize Grad-CAM to match image dimensions
            if gradcam_heatmap.shape != (h, w):
                gc_tensor = torch.from_numpy(gradcam_heatmap).unsqueeze(0).unsqueeze(0)
                gc_resized = F.interpolate(gc_tensor, size=(h, w), mode='bilinear', align_corners=False)
                gradcam_heatmap = gc_resized.squeeze().numpy()

            combined_heatmap = (
                0.4 * ela_map +
                0.4 * gradcam_heatmap +
                0.2 * stat_map
            )
        else:
            combined_heatmap = (
                0.6 * ela_map +
                0.4 * stat_map
            )

        # Normalize combined heatmap
        combined_heatmap = np.clip(combined_heatmap, 0, 1)
        if combined_heatmap.max() > 0:
            combined_heatmap = combined_heatmap / combined_heatmap.max()

        # === Generate Binary Mask ===
        mask = combined_heatmap > self.threshold

        # === Extract Bounding Boxes ===
        regions = self._extract_regions(mask, combined_heatmap, h, w)

        # === Compute Overall Manipulation Score ===
        manipulation_score = self._compute_manipulation_score(
            ela_map, combined_heatmap, mask
        )

        result = ForgeryLocalizationResult(
            heatmap=combined_heatmap,
            regions=regions,
            mask=mask,
            manipulation_score=manipulation_score,
            ela_map=ela_map,
        )

        logger.info(
            f"Forgery localization: score={manipulation_score:.3f}, "
            f"regions={len(regions)}"
        )

        return result

    @staticmethod
    def _compute_statistical_map(image: np.ndarray) -> np.ndarray:
        """
        Compute pixel-level statistical deviation map.

        Identifies regions that deviate significantly from local statistics,
        which may indicate splicing or inpainting.

        Args:
            image: Normalized image array [H, W, 3].

        Returns:
            Deviation map [H, W], values in [0, 1].
        """
        grayscale = image.mean(axis=2)  # [H, W]

        # Local mean using block processing
        from PIL import Image as PILImage, ImageFilter

        # Convert to PIL for efficient filtering
        pil_img = PILImage.fromarray((grayscale * 255).astype(np.uint8), 'L')

        # Local mean (blur approximation)
        local_mean = np.array(pil_img.filter(ImageFilter.BoxBlur(15)), dtype=np.float32) / 255.0

        # Deviation from local mean
        deviation = np.abs(grayscale - local_mean)

        # Normalize
        if deviation.max() > 0:
            deviation = deviation / deviation.max()

        return deviation

    def _extract_regions(
        self,
        mask: np.ndarray,
        heatmap: np.ndarray,
        img_h: int,
        img_w: int
    ) -> List[ForgeryRegion]:
        """
        Extract suspicious regions from binary mask using connected components.

        Args:
            mask: Binary mask [H, W].
            heatmap: Continuous heatmap [H, W].
            img_h: Image height.
            img_w: Image width.

        Returns:
            List of ForgeryRegion objects.
        """
        regions = []

        try:
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(mask)

            total_area = img_h * img_w

            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                area = region_mask.sum()

                if area < self.min_region_area:
                    continue

                # Get bounding box
                rows = np.where(region_mask.any(axis=1))[0]
                cols = np.where(region_mask.any(axis=0))[0]

                if len(rows) == 0 or len(cols) == 0:
                    continue

                bbox = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
                confidence = float(heatmap[region_mask].mean())
                area_ratio = float(area / total_area)

                regions.append(ForgeryRegion(
                    bbox=bbox,
                    confidence=confidence,
                    area_ratio=area_ratio,
                    label=f"Suspicious Region #{len(regions) + 1}"
                ))

        except ImportError:
            logger.warning("scipy not available, using simplified region extraction")
            # Fallback: use the entire mask as one region if significant
            if mask.sum() > self.min_region_area:
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                if len(rows) > 0 and len(cols) > 0:
                    regions.append(ForgeryRegion(
                        bbox=(int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])),
                        confidence=float(heatmap[mask].mean()),
                        area_ratio=float(mask.sum() / (img_h * img_w)),
                        label="Suspicious Region (combined)"
                    ))

        return regions

    @staticmethod
    def _compute_manipulation_score(
        ela_map: np.ndarray,
        heatmap: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Compute overall manipulation likelihood score.

        Factors:
            - Mean ELA intensity (high → likely manipulated)
            - Heatmap variance (non-uniform → suspicious)
            - Mask coverage ratio
        """
        ela_mean = float(ela_map.mean())
        heatmap_std = float(heatmap.std())
        mask_ratio = float(mask.sum() / max(mask.size, 1))

        score = (
            0.4 * min(ela_mean * 3.0, 1.0) +
            0.3 * min(heatmap_std * 4.0, 1.0) +
            0.3 * min(mask_ratio * 5.0, 1.0)
        )

        return min(1.0, max(0.0, score))
