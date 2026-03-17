"""
TrueLens AI - Metadata Analysis
Analyzes image metadata (EXIF, IPTC, XMP) to detect signs of AI generation.

AI-generated images typically lack camera-specific metadata that natural
photographs contain, such as camera model, lens info, GPS coordinates,
and exposure settings.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import json


class MetadataAnalyzer:
    """
    Analyzes image metadata to determine authenticity.
    
    Real photographs from cameras contain rich EXIF metadata including:
    - Camera make/model
    - Lens information
    - Exposure settings (ISO, shutter speed, aperture)
    - GPS coordinates
    - Timestamps
    
    AI-generated images typically lack most or all of this metadata,
    or contain suspicious metadata patterns.
    """
    
    # Key EXIF tags that indicate a real camera capture
    CAMERA_TAGS = [
        'Make', 'Model', 'LensModel', 'LensMake',
        'BodySerialNumber', 'LensSerialNumber'
    ]
    
    EXPOSURE_TAGS = [
        'ExposureTime', 'FNumber', 'ISOSpeedRatings', 'ISO',
        'ExposureProgram', 'ExposureBiasValue', 'MeteringMode',
        'Flash', 'FocalLength', 'FocalLengthIn35mmFilm'
    ]
    
    GPS_TAGS = [
        'GPSLatitude', 'GPSLongitude', 'GPSAltitude',
        'GPSLatitudeRef', 'GPSLongitudeRef'
    ]
    
    SOFTWARE_KEYWORDS = [
        'photoshop', 'gimp', 'lightroom', 'capture one',
        'darktable', 'rawtherapee', 'affinity',
        'midjourney', 'dall-e', 'stable diffusion', 'comfyui',
        'automatic1111', 'invoke ai', 'novelai'
    ]
    
    AI_SOFTWARE_KEYWORDS = [
        'midjourney', 'dall-e', 'stable diffusion', 'comfyui',
        'automatic1111', 'invoke ai', 'novelai', 'leonardo',
        'firefly', 'bing image creator', 'copilot',
        'ai generated', 'ai-generated',
        'gemini', 'google ai', 'imagen', 'deepmind',
        'chatgpt', 'openai', 'anthropic', 'claude',
        'flux', 'ideogram', 'playground', 'craiyon',
        'nightcafe', 'artbreeder', 'dream studio',
        'stability ai', 'runway', 'pika'
    ]
    
    def analyze(self, image: Image.Image) -> dict:
        """
        Perform comprehensive metadata analysis.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with metadata analysis results
        """
        metadata = self._extract_metadata(image)
        
        # Analyze various aspects
        camera_info = self._analyze_camera_info(metadata)
        exposure_info = self._analyze_exposure_info(metadata)
        gps_info = self._analyze_gps_info(metadata)
        software_info = self._analyze_software_info(metadata)
        
        # Compute metadata authenticity score
        metadata_score = self._compute_metadata_score(
            camera_info, exposure_info, gps_info, software_info, metadata
        )
        
        # Generate analysis text
        analysis_text = self._generate_analysis_text(
            metadata_score, camera_info, software_info, metadata
        )
        
        return {
            "has_metadata": len(metadata) > 0,
            "metadata_count": len(metadata),
            "camera_info": camera_info,
            "exposure_info": exposure_info,
            "has_gps": gps_info["has_gps"],
            "software_info": software_info,
            "metadata_score": round(metadata_score, 4),
            "analysis_text": analysis_text,
            "raw_metadata": {k: str(v) for k, v in list(metadata.items())[:20]}
        }
    
    def _extract_metadata(self, image: Image.Image) -> dict:
        """Extract all available metadata from the image."""
        metadata = {}
        
        try:
            raw_exif = image._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    try:
                        if tag_name == 'GPSInfo':
                            gps_data = {}
                            for gps_tag_id, gps_value in value.items():
                                gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                                gps_data[gps_tag_name] = gps_value
                            metadata['GPSInfo'] = gps_data
                        else:
                            metadata[tag_name] = value
                    except Exception:
                        metadata[tag_name] = str(value)
        except Exception:
            pass
        
        # Also check image.info for additional metadata
        try:
            for key, value in image.info.items():
                if key not in metadata:
                    metadata[key] = value
        except Exception:
            pass
        
        return metadata
    
    def _analyze_camera_info(self, metadata: dict) -> dict:
        """Analyze camera-related metadata."""
        found_tags = {}
        for tag in self.CAMERA_TAGS:
            if tag in metadata:
                found_tags[tag] = str(metadata[tag])
        
        return {
            "has_camera_info": len(found_tags) > 0,
            "camera_tags_found": len(found_tags),
            "camera_make": found_tags.get('Make', 'Unknown'),
            "camera_model": found_tags.get('Model', 'Unknown'),
            "lens_model": found_tags.get('LensModel', 'Unknown'),
            "details": found_tags
        }
    
    def _analyze_exposure_info(self, metadata: dict) -> dict:
        """Analyze exposure-related metadata."""
        found_tags = {}
        for tag in self.EXPOSURE_TAGS:
            if tag in metadata:
                found_tags[tag] = str(metadata[tag])
        
        return {
            "has_exposure_info": len(found_tags) > 0,
            "exposure_tags_found": len(found_tags),
            "details": found_tags
        }
    
    def _analyze_gps_info(self, metadata: dict) -> dict:
        """Analyze GPS metadata."""
        gps_data = metadata.get('GPSInfo', {})
        
        has_gps = isinstance(gps_data, dict) and len(gps_data) > 0
        
        return {
            "has_gps": has_gps,
            "gps_tags_found": len(gps_data) if isinstance(gps_data, dict) else 0
        }
    
    def _analyze_software_info(self, metadata: dict) -> dict:
        """Analyze software-related metadata for AI generation clues."""
        software = str(metadata.get('Software', '')).lower()
        processing_software = str(metadata.get('ProcessingSoftware', '')).lower()
        
        all_software = f"{software} {processing_software}"
        
        # Check for AI generation software
        is_ai_tagged = any(kw in all_software for kw in self.AI_SOFTWARE_KEYWORDS)
        
        # Check for editing software
        is_edited = any(kw in all_software for kw in self.SOFTWARE_KEYWORDS)
        
        # Check for AI-related metadata fields
        ai_parameters = {}
        for key in ['parameters', 'prompt', 'negative_prompt', 'sd-metadata',
                     'Dream', 'Comment', 'UserComment']:
            if key in metadata:
                value = str(metadata[key])[:200]  # Truncate long values
                if any(kw in value.lower() for kw in ['prompt', 'steps', 'sampler', 'cfg', 'seed']):
                    ai_parameters[key] = value
                    is_ai_tagged = True
        
        return {
            "software": metadata.get('Software', 'None detected'),
            "is_ai_tagged": is_ai_tagged,
            "is_edited": is_edited,
            "ai_parameters_found": len(ai_parameters) > 0,
            "ai_parameters": ai_parameters
        }
    
    def _compute_metadata_score(self, camera_info, exposure_info, gps_info, software_info, metadata):
        """
        Compute metadata-based AI detection score.
        
        Returns:
            Score from 0 (likely real) to 1 (likely AI-generated)
            
        AI-generated images from modern generators (Gemini, DALL-E, 
        Stable Diffusion, Midjourney) almost NEVER contain real camera 
        metadata. This is one of the strongest forensic indicators.
        """
        score = 0.5  # Start neutral
        
        # AI software detected - definitive indicator
        if software_info["is_ai_tagged"]:
            return 0.98
        
        if software_info["ai_parameters_found"]:
            return 0.98
        
        # ---- No metadata at all = very strong AI indicator ----
        if len(metadata) == 0:
            return 0.85  # Extremely suspicious
        
        # Very few metadata tags (1-3) = still very suspicious
        if len(metadata) <= 3:
            score += 0.25
        
        # ---- Camera info ----
        if camera_info["has_camera_info"]:
            score -= 0.30  # Strong evidence of real camera
            if camera_info["camera_tags_found"] >= 3:
                score -= 0.1  # Even stronger (make, model, lens)
        else:
            score += 0.20  # No camera = suspicious
        
        # ---- Exposure info ----
        if exposure_info["has_exposure_info"]:
            score -= 0.15
            if exposure_info["exposure_tags_found"] >= 4:
                score -= 0.1  # Rich exposure data = almost certainly real
        else:
            score += 0.15  # No exposure = suspicious
        
        # ---- GPS present ----
        if gps_info["has_gps"]:
            score -= 0.2  # GPS data = almost certainly from a real device
        
        # ---- Metadata richness ----
        if len(metadata) > 20:
            score -= 0.1  # Rich metadata = likely real
        elif len(metadata) > 10:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def _generate_analysis_text(self, score, camera_info, software_info, metadata):
        """Generate human-readable metadata analysis text."""
        parts = []
        
        if software_info["is_ai_tagged"]:
            parts.append("🚨 AI GENERATION SOFTWARE DETECTED in image metadata!")
            if software_info["ai_parameters_found"]:
                parts.append("AI generation parameters (prompt, seed, etc.) found embedded in the file.")
            return " ".join(parts)
        
        if score > 0.6:
            parts.append("⚠️ Metadata analysis suggests this image may be AI-generated.")
        elif score > 0.4:
            parts.append("⚡ Metadata provides inconclusive evidence about image origin.")
        else:
            parts.append("✅ Metadata is consistent with a camera-captured photograph.")
        
        if camera_info["has_camera_info"]:
            make = camera_info["camera_make"]
            model = camera_info["camera_model"]
            parts.append(f"Camera: {make} {model}.")
        else:
            parts.append("No camera information found in metadata.")
        
        if len(metadata) == 0:
            parts.append("Image contains no EXIF metadata, which is unusual for camera photos but common for AI-generated images.")
        else:
            parts.append(f"Found {len(metadata)} metadata tags.")
        
        return " ".join(parts)
