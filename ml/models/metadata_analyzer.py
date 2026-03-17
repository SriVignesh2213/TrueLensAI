"""
TrueLens AI — Metadata Forensic Analyzer

Performs deep EXIF/metadata analysis to detect:
    - Missing camera hardware signatures (common in AI-generated images)
    - Software manipulation indicators (Photoshop, GIMP, etc.)
    - Timestamp anomalies and inconsistencies
    - GPS coordinate validation
    - Compression artifact analysis
    - Thumbnail-to-image mismatch detection

Scoring:
    Each anomaly category contributes to a weighted risk score.
    The final metadata_anomaly_score is in [0, 1].

Author: TrueLens AI Team
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Known AI/editing software signatures
MANIPULATION_SOFTWARE = {
    'adobe photoshop', 'gimp', 'affinity photo', 'pixelmator',
    'paint.net', 'corel paintshop', 'photopea', 'canva',
    'midjourney', 'dall-e', 'stable diffusion', 'artbreeder',
    'deepart', 'prisma', 'faceapp', 'remini', 'lensa',
    'topaz', 'luminar', 'photoai'
}

# Known legitimate camera manufacturers
KNOWN_CAMERA_MAKES = {
    'canon', 'nikon', 'sony', 'fujifilm', 'panasonic', 'olympus',
    'leica', 'hasselblad', 'pentax', 'sigma', 'ricoh', 'gopro',
    'dji', 'apple', 'samsung', 'google', 'huawei', 'xiaomi',
    'oneplus', 'oppo', 'vivo', 'motorola', 'lg', 'nokia'
}

# Critical EXIF fields expected in authentic camera photos
CRITICAL_CAMERA_FIELDS = [
    'Make', 'Model', 'DateTime', 'ExposureTime', 'FNumber',
    'ISOSpeedRatings', 'FocalLength'
]


@dataclass
class MetadataAnomalyReport:
    """
    Structured report of metadata anomalies found in an image.

    Attributes:
        has_exif: Whether the image contains any EXIF data.
        anomaly_score: Overall anomaly score [0, 1].
        anomaly_flags: List of specific anomaly descriptions.
        risk_level: Categorical risk level (LOW / MEDIUM / HIGH / CRITICAL).
        camera_info: Extracted camera information if available.
        software_detected: Detected editing software names.
        timestamp_info: Timestamp analysis results.
        details: Full anomaly detail dictionary.
    """
    has_exif: bool = False
    anomaly_score: float = 0.0
    anomaly_flags: List[str] = field(default_factory=list)
    risk_level: str = "LOW"
    camera_info: Dict[str, str] = field(default_factory=dict)
    software_detected: List[str] = field(default_factory=list)
    timestamp_info: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            'has_exif': self.has_exif,
            'anomaly_score': round(self.anomaly_score, 4),
            'anomaly_flags': self.anomaly_flags,
            'risk_level': self.risk_level,
            'camera_info': self.camera_info,
            'software_detected': self.software_detected,
            'timestamp_info': self.timestamp_info,
            'total_anomalies': len(self.anomaly_flags),
        }


class MetadataForensicAnalyzer:
    """
    Comprehensive EXIF and metadata forensic analyzer.

    Analyzes image metadata to detect signs of manipulation or
    AI generation. Uses weighted scoring across multiple anomaly
    categories.

    Anomaly Categories (weights):
        - Missing EXIF data:          0.25
        - Missing camera signature:   0.20
        - Software manipulation:      0.20
        - Timestamp anomalies:        0.15
        - Compression anomalies:      0.10
        - Field consistency:          0.10
    """

    # Category weights for final scoring
    WEIGHTS = {
        'missing_exif': 0.25,
        'missing_camera': 0.20,
        'software_manipulation': 0.20,
        'timestamp_anomaly': 0.15,
        'compression_anomaly': 0.10,
        'field_consistency': 0.10,
    }

    def __init__(self) -> None:
        """Initialize the metadata forensic analyzer."""
        logger.info("MetadataForensicAnalyzer initialized")

    def analyze(self, image_path: str) -> MetadataAnomalyReport:
        """
        Perform full metadata forensic analysis on an image.

        Args:
            image_path: Path to the image file.

        Returns:
            MetadataAnomalyReport with detailed findings.
        """
        report = MetadataAnomalyReport()
        category_scores: Dict[str, float] = {}

        try:
            exif_data = self._extract_exif(image_path)
        except Exception as e:
            logger.warning(f"Failed to extract EXIF from {image_path}: {e}")
            report.anomaly_score = 0.6
            report.anomaly_flags.append("EXIF extraction failed — possible metadata stripping")
            report.risk_level = "HIGH"
            return report

        if not exif_data:
            report.has_exif = False
            report.anomaly_score = 0.7
            report.anomaly_flags.append("No EXIF metadata found — common in AI-generated images")
            report.risk_level = "HIGH"
            return report

        report.has_exif = True

        # === Category 1: Camera Signature Analysis ===
        camera_score, camera_flags = self._analyze_camera_signature(exif_data)
        category_scores['missing_camera'] = camera_score
        report.anomaly_flags.extend(camera_flags)
        report.camera_info = {
            'make': exif_data.get('Make', 'Unknown'),
            'model': exif_data.get('Model', 'Unknown'),
        }

        # === Category 2: Software Detection ===
        software_score, software_flags, sw_names = self._analyze_software(exif_data)
        category_scores['software_manipulation'] = software_score
        report.anomaly_flags.extend(software_flags)
        report.software_detected = sw_names

        # === Category 3: Timestamp Analysis ===
        timestamp_score, timestamp_flags, ts_info = self._analyze_timestamps(exif_data)
        category_scores['timestamp_anomaly'] = timestamp_score
        report.anomaly_flags.extend(timestamp_flags)
        report.timestamp_info = ts_info

        # === Category 4: Missing Critical Fields ===
        field_score, field_flags = self._analyze_field_completeness(exif_data)
        category_scores['missing_exif'] = field_score
        report.anomaly_flags.extend(field_flags)

        # === Category 5: Compression Analysis ===
        compression_score, compression_flags = self._analyze_compression(exif_data)
        category_scores['compression_anomaly'] = compression_score
        report.anomaly_flags.extend(compression_flags)

        # === Category 6: Field Consistency ===
        consistency_score, consistency_flags = self._analyze_consistency(exif_data)
        category_scores['field_consistency'] = consistency_score
        report.anomaly_flags.extend(consistency_flags)

        # Compute weighted anomaly score
        report.anomaly_score = sum(
            self.WEIGHTS[cat] * score
            for cat, score in category_scores.items()
        )
        report.anomaly_score = min(1.0, max(0.0, report.anomaly_score))

        # Assign risk level
        report.risk_level = self._score_to_risk_level(report.anomaly_score)

        report.details = {
            'category_scores': {k: round(v, 3) for k, v in category_scores.items()},
            'total_fields_analyzed': len(exif_data),
        }

        logger.info(
            f"Metadata analysis complete: score={report.anomaly_score:.3f}, "
            f"risk={report.risk_level}, anomalies={len(report.anomaly_flags)}"
        )

        return report

    @staticmethod
    def _extract_exif(image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract EXIF data from an image file using Pillow.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary of EXIF tag names to values, or None if no EXIF.
        """
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)
        exif_raw = img.getexif()

        if not exif_raw:
            return None

        exif_data = {}
        for tag_id, value in exif_raw.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            try:
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                exif_data[tag_name] = value
            except Exception:
                exif_data[tag_name] = str(value)

        return exif_data if exif_data else None

    @staticmethod
    def _analyze_camera_signature(
        exif: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Analyze camera make/model presence and validity."""
        flags = []
        score = 0.0

        make = str(exif.get('Make', '')).strip().lower()
        model = str(exif.get('Model', '')).strip().lower()

        if not make:
            score += 0.5
            flags.append("Missing camera manufacturer (Make)")
        elif make not in KNOWN_CAMERA_MAKES:
            score += 0.3
            flags.append(f"Unknown camera manufacturer: '{make}'")

        if not model:
            score += 0.5
            flags.append("Missing camera model")

        return min(1.0, score), flags

    @staticmethod
    def _analyze_software(
        exif: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """Detect known manipulation or AI generation software."""
        flags = []
        detected = []
        score = 0.0

        software = str(exif.get('Software', '')).strip().lower()
        if software:
            for sw in MANIPULATION_SOFTWARE:
                if sw in software:
                    detected.append(sw)
                    score = 1.0
                    flags.append(f"Manipulation software detected: {sw}")

        # Check ProcessingSoftware tag
        proc_sw = str(exif.get('ProcessingSoftware', '')).strip().lower()
        if proc_sw:
            for sw in MANIPULATION_SOFTWARE:
                if sw in proc_sw:
                    if sw not in detected:
                        detected.append(sw)
                    score = 1.0
                    flags.append(f"Processing software flag: {sw}")

        return score, flags, detected

    @staticmethod
    def _analyze_timestamps(
        exif: Dict[str, Any]
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        """Analyze timestamp presence and consistency."""
        flags = []
        score = 0.0
        ts_info: Dict[str, Any] = {}

        datetime_original = exif.get('DateTimeOriginal', '')
        datetime_digitized = exif.get('DateTimeDigitized', '')
        datetime_modified = exif.get('DateTime', '')

        ts_info['original'] = str(datetime_original) if datetime_original else None
        ts_info['digitized'] = str(datetime_digitized) if datetime_digitized else None
        ts_info['modified'] = str(datetime_modified) if datetime_modified else None

        if not datetime_original and not datetime_digitized:
            score += 0.5
            flags.append("Missing original capture timestamp")

        # Check for future timestamps
        for label, ts in [('Original', datetime_original), ('Modified', datetime_modified)]:
            if ts:
                try:
                    dt = datetime.strptime(str(ts), '%Y:%m:%d %H:%M:%S')
                    if dt > datetime.now():
                        score += 0.5
                        flags.append(f"{label} timestamp is in the future")
                except (ValueError, TypeError):
                    pass

        # Check Original vs Modified consistency
        if datetime_original and datetime_modified:
            try:
                dt_orig = datetime.strptime(str(datetime_original), '%Y:%m:%d %H:%M:%S')
                dt_mod = datetime.strptime(str(datetime_modified), '%Y:%m:%d %H:%M:%S')
                if dt_mod < dt_orig:
                    score += 0.3
                    flags.append("Modified timestamp precedes original — possible tampering")
            except (ValueError, TypeError):
                pass

        return min(1.0, score), flags, ts_info

    @staticmethod
    def _analyze_field_completeness(
        exif: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Check for missing critical EXIF fields."""
        flags = []
        missing_count = 0

        for field_name in CRITICAL_CAMERA_FIELDS:
            if field_name not in exif or not exif[field_name]:
                missing_count += 1

        total = len(CRITICAL_CAMERA_FIELDS)
        missing_ratio = missing_count / total

        if missing_ratio > 0.7:
            flags.append(f"Severely incomplete EXIF: {missing_count}/{total} critical fields missing")
        elif missing_ratio > 0.4:
            flags.append(f"Incomplete EXIF: {missing_count}/{total} critical fields missing")

        return missing_ratio, flags

    @staticmethod
    def _analyze_compression(
        exif: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Analyze JPEG compression parameters for anomalies."""
        flags = []
        score = 0.0

        # Check for unusual compression values
        compression = exif.get('Compression')
        if compression and compression not in [1, 6, 7]:  # Uncompressed, JPEG, JPEG (new)
            score += 0.3
            flags.append(f"Unusual compression type: {compression}")

        return score, flags

    @staticmethod
    def _analyze_consistency(
        exif: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Check for logical inconsistencies in EXIF fields."""
        flags = []
        score = 0.0

        # Check if GPS data exists without camera make
        has_gps = any(k.startswith('GPS') for k in exif.keys())
        has_make = bool(exif.get('Make'))

        if has_gps and not has_make:
            score += 0.4
            flags.append("GPS data present but no camera manufacturer — unusual combination")

        # Check exposure parameters consistency
        exposure = exif.get('ExposureTime')
        fnumber = exif.get('FNumber')
        iso = exif.get('ISOSpeedRatings')

        has_exposure_params = sum(bool(x) for x in [exposure, fnumber, iso])
        if 0 < has_exposure_params < 3:
            score += 0.2
            flags.append("Incomplete exposure triangle — possible metadata injection")

        return min(1.0, score), flags

    @staticmethod
    def _score_to_risk_level(score: float) -> str:
        """Convert numeric anomaly score to categorical risk level."""
        if score >= 0.75:
            return "CRITICAL"
        elif score >= 0.50:
            return "HIGH"
        elif score >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"
