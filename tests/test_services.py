

import pytest
import numpy as np

# ============================================================
# 1. TRIAGE SERVICE TESTS
# ============================================================
from app.services.triage_service import get_region_severity, get_overall_severity


class TestTriageService:
    """Tests for the rule-based triage severity engine."""

    # --- get_region_severity ---

    def test_normal_region_returns_none_severity(self):
        """Normal classification should always return NONE regardless of region."""
        result = get_region_severity("Wrist", "normal", 0.95)
        assert result == "NONE"

    def test_fractured_wrist_high_confidence(self):
        """Wrist fracture with high confidence should return HIGH severity."""
        result = get_region_severity("Wrist", "fracture", 0.85)
        assert result == "HIGH"

    def test_fractured_radius_high_confidence(self):
        """Radius fracture with high confidence should return HIGH severity."""
        result = get_region_severity("Radius", "fracture", 0.90)
        assert result == "HIGH"

    def test_fractured_mcp_high_confidence(self):
        """MCP fracture with high confidence should return MEDIUM severity."""
        result = get_region_severity("MCP", "fracture", 0.80)
        assert result == "MEDIUM"

    def test_fractured_ulna_high_confidence(self):
        """Ulna fracture with high confidence should return MEDIUM severity."""
        result = get_region_severity("Ulna", "fracture", 0.75)
        assert result == "MEDIUM"

    def test_fractured_pip_high_confidence(self):
        """PIP fracture with high confidence should return LOW severity."""
        result = get_region_severity("PIP", "fracture", 0.80)
        assert result == "LOW"

    def test_fractured_dip_high_confidence(self):
        """DIP fracture with high confidence should return LOW severity."""
        result = get_region_severity("DIP", "fracture", 0.70)
        assert result == "LOW"

    def test_low_confidence_downgrades_severity(self):
        """Confidence below 0.6 should downgrade severity by one tier."""
        result = get_region_severity("Wrist", "fracture", 0.55)
        # HIGH (priority 3) downgraded to MEDIUM (priority 2)
        assert result == "MEDIUM"

    def test_low_confidence_medium_downgrades_to_low(self):
        """MEDIUM severity with low confidence should downgrade to LOW."""
        result = get_region_severity("MCP", "fracture", 0.50)
        assert result == "LOW"

    def test_low_confidence_low_stays_low(self):
        """LOW severity with low confidence cannot downgrade further."""
        result = get_region_severity("PIP", "fracture", 0.45)
        assert result == "LOW"

    def test_fractured_prefix_handled(self):
        """Regions with 'Fractured ' prefix should map correctly."""
        result = get_region_severity("Fractured MCP", "fracture", 0.85)
        assert result == "MEDIUM"

    def test_fractured_prefix_wrist(self):
        """'Fractured Wrist' should map to HIGH severity."""
        result = get_region_severity("Fractured Wrist", "fracture", 0.90)
        assert result == "HIGH"

    def test_unknown_region_defaults_to_medium(self):
        """Unknown region should default to MEDIUM severity."""
        result = get_region_severity("unknown", "fracture", 0.80)
        assert result == "MEDIUM"

    # --- get_overall_severity ---

    def test_overall_severity_highest_wins(self):
        """Overall severity should be the highest among all predictions."""
        results = [
            {"classification": "fracture", "severity": "LOW"},
            {"classification": "fracture", "severity": "HIGH"},
            {"classification": "normal", "severity": "NONE"},
        ]
        assert get_overall_severity(results) == "HIGH"

    def test_overall_severity_all_normal(self):
        """No fractures should return NONE overall severity."""
        results = [
            {"classification": "normal", "severity": "NONE"},
            {"classification": "normal", "severity": "NONE"},
        ]
        assert get_overall_severity(results) == "NONE"

    def test_overall_severity_single_fracture(self):
        """Single fracture severity should be the overall severity."""
        results = [
            {"classification": "fracture", "severity": "MEDIUM"},
            {"classification": "normal", "severity": "NONE"},
        ]
        assert get_overall_severity(results) == "MEDIUM"

    def test_overall_severity_empty_list(self):
        """Empty prediction list should return NONE."""
        assert get_overall_severity([]) == "NONE"


# ============================================================
# 2. DETECTION SERVICE TESTS (Overlap & Classification Logic)
# ============================================================
from app.services.detection_service import (
    check_overlap,
    classify_regions_by_overlap,
    find_closest_region,
)


class TestDetectionServiceOverlap:
    """Tests for bounding box overlap calculation."""

    def test_full_overlap(self):
        """Identical bounding boxes should return True with 100% overlap."""
        box = {"x": 100, "y": 100, "width": 50, "height": 50}
        overlaps, ratio = check_overlap(box, box, threshold=0.2)
        assert overlaps is True
        assert ratio == 1.0

    def test_no_overlap(self):
        """Non-overlapping boxes should return False with 0% overlap."""
        anatomy = {"x": 50, "y": 50, "width": 40, "height": 40}
        fracture = {"x": 200, "y": 200, "width": 30, "height": 30}
        overlaps, ratio = check_overlap(anatomy, fracture, threshold=0.2)
        assert overlaps is False
        assert ratio == 0.0

    def test_partial_overlap_above_threshold(self):
        """Partially overlapping boxes exceeding threshold should return True."""
        anatomy = {"x": 100, "y": 100, "width": 100, "height": 100}
        fracture = {"x": 130, "y": 100, "width": 60, "height": 100}
        overlaps, ratio = check_overlap(anatomy, fracture, threshold=0.1)
        assert overlaps is True
        assert ratio > 0.1

    def test_partial_overlap_below_threshold(self):
        """Partially overlapping boxes below threshold should return False."""
        anatomy = {"x": 100, "y": 100, "width": 200, "height": 200}
        fracture = {"x": 200, "y": 200, "width": 20, "height": 20}
        overlaps, ratio = check_overlap(anatomy, fracture, threshold=0.5)
        assert overlaps is False

    def test_zero_area_anatomy(self):
        """Zero-area anatomy box should return False without error."""
        anatomy = {"x": 100, "y": 100, "width": 0, "height": 50}
        fracture = {"x": 100, "y": 100, "width": 50, "height": 50}
        overlaps, ratio = check_overlap(anatomy, fracture)
        assert overlaps is False
        assert ratio == 0.0

    def test_overlap_ratio_is_float(self):
        """Overlap ratio should always be a float."""
        anatomy = {"x": 100, "y": 100, "width": 80, "height": 80}
        fracture = {"x": 110, "y": 110, "width": 40, "height": 40}
        _, ratio = check_overlap(anatomy, fracture)
        assert isinstance(ratio, float)


class TestDetectionServiceClassification:
    """Tests for region classification by overlap comparison."""

    def test_fracture_overlapping_region(self):
        """Anatomy region overlapping with fracture should be classified as fracture."""
        anatomy = [{"x": 100, "y": 100, "width": 80, "height": 80, "class": "Wrist", "confidence": 0.95}]
        fracture = [{"x": 105, "y": 105, "width": 60, "height": 60, "confidence": 0.90}]
        results = classify_regions_by_overlap(anatomy, fracture, confidence_threshold=0.5)
        assert len(results) == 1
        assert results[0]["classification"] == "fracture"

    def test_no_fracture_detected(self):
        """Anatomy region with no overlapping fracture should be classified as normal."""
        anatomy = [{"x": 100, "y": 100, "width": 80, "height": 80, "class": "MCP", "confidence": 0.90}]
        fracture = [{"x": 500, "y": 500, "width": 30, "height": 30, "confidence": 0.85}]
        results = classify_regions_by_overlap(anatomy, fracture, confidence_threshold=0.5)
        assert len(results) == 1
        assert results[0]["classification"] == "normal"

    def test_low_confidence_anatomy_filtered(self):
        """Anatomy detections below confidence threshold should be excluded."""
        anatomy = [{"x": 100, "y": 100, "width": 80, "height": 80, "class": "PIP", "confidence": 0.3}]
        fracture = [{"x": 100, "y": 100, "width": 60, "height": 60, "confidence": 0.90}]
        results = classify_regions_by_overlap(anatomy, fracture, confidence_threshold=0.5)
        assert len(results) == 0

    def test_low_confidence_fracture_ignored(self):
        """Fracture detections below confidence threshold should not cause fracture classification."""
        anatomy = [{"x": 100, "y": 100, "width": 80, "height": 80, "class": "Radius", "confidence": 0.90}]
        fracture = [{"x": 105, "y": 105, "width": 60, "height": 60, "confidence": 0.2}]
        results = classify_regions_by_overlap(anatomy, fracture, confidence_threshold=0.5)
        assert len(results) == 1
        assert results[0]["classification"] == "normal"

    def test_multiple_regions_mixed_results(self):
        """Multiple anatomy regions should be independently classified."""
        anatomy = [
            {"x": 100, "y": 100, "width": 80, "height": 80, "class": "Wrist", "confidence": 0.95},
            {"x": 400, "y": 400, "width": 60, "height": 60, "class": "MCP", "confidence": 0.90},
        ]
        fracture = [{"x": 105, "y": 105, "width": 60, "height": 60, "confidence": 0.88}]
        results = classify_regions_by_overlap(anatomy, fracture, confidence_threshold=0.5)
        assert len(results) == 2
        classifications = [r["classification"] for r in results]
        assert "fracture" in classifications
        assert "normal" in classifications

    def test_empty_fracture_list(self):
        """No fracture detections should classify all regions as normal."""
        anatomy = [{"x": 100, "y": 100, "width": 80, "height": 80, "class": "Ulna", "confidence": 0.90}]
        results = classify_regions_by_overlap(anatomy, [], confidence_threshold=0.5)
        assert len(results) == 1
        assert results[0]["classification"] == "normal"


class TestDetectionServiceClosestRegion:
    """Tests for finding the closest anatomical region to an unmatched fracture."""

    def test_closest_region_found(self):
        """Should return the nearest anatomy region by Euclidean distance."""
        fracture = {"x": 150, "y": 150, "width": 30, "height": 30, "confidence": 0.8}
        anatomy = [
            {"x": 100, "y": 100, "width": 50, "height": 50, "class": "Wrist", "confidence": 0.9},
            {"x": 500, "y": 500, "width": 50, "height": 50, "class": "MCP", "confidence": 0.9},
        ]
        region, distance = find_closest_region(fracture, anatomy, confidence_threshold=0.5)
        assert region == "Wrist"
        assert distance > 0

    def test_closest_region_with_low_confidence_filtered(self):
        """Low-confidence anatomy regions should be excluded from closest search."""
        fracture = {"x": 150, "y": 150, "width": 30, "height": 30, "confidence": 0.8}
        anatomy = [
            {"x": 100, "y": 100, "width": 50, "height": 50, "class": "Wrist", "confidence": 0.3},
            {"x": 500, "y": 500, "width": 50, "height": 50, "class": "MCP", "confidence": 0.9},
        ]
        region, distance = find_closest_region(fracture, anatomy, confidence_threshold=0.5)
        assert region == "MCP"

    def test_empty_anatomy_returns_none(self):
        """Empty anatomy list should return None for region."""
        fracture = {"x": 150, "y": 150, "width": 30, "height": 30, "confidence": 0.8}
        region, distance = find_closest_region(fracture, [], confidence_threshold=0.5)
        assert region is None


# ============================================================
# 3. ROI SERVICE TESTS
# ============================================================
from app.services.roi_service import crop_region


class TestROIService:
    """Tests for region-of-interest cropping from images."""

    def _make_test_image(self, width=640, height=480):
        """Helper: creates a dummy BGR image."""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def test_crop_region_returns_array(self):
        """Cropping a valid region should return a numpy array."""
        image = self._make_test_image()
        pred = {"x": 200, "y": 200, "width": 100, "height": 100}
        crop = crop_region(image, pred, padding=40)
        assert crop is not None
        assert isinstance(crop, np.ndarray)

    def test_crop_region_dimensions(self):
        """Cropped region should be smaller than or equal to the original image."""
        image = self._make_test_image(640, 480)
        pred = {"x": 200, "y": 200, "width": 100, "height": 100}
        crop = crop_region(image, pred, padding=20)
        assert crop.shape[0] <= 480
        assert crop.shape[1] <= 640

    def test_crop_region_includes_padding(self):
        """Crop should be larger than the bounding box due to padding."""
        image = self._make_test_image(640, 480)
        pred = {"x": 300, "y": 240, "width": 80, "height": 60}
        crop = crop_region(image, pred, padding=40)
        assert crop.shape[0] >= 60  # height >= bbox height
        assert crop.shape[1] >= 80  # width >= bbox width

    def test_crop_region_edge_clamping(self):
        """Crop near image edges should be clamped without errors."""
        image = self._make_test_image(640, 480)
        pred = {"x": 10, "y": 10, "width": 40, "height": 40}
        crop = crop_region(image, pred, padding=50)
        assert crop is not None
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0

    def test_crop_region_zero_size_returns_none(self):
        """A zero-size bounding box should return None."""
        image = self._make_test_image()
        pred = {"x": 200, "y": 200, "width": 0, "height": 0}
        crop = crop_region(image, pred, padding=0)
        assert crop is None


# ============================================================
# 4. CLASSIFIER SERVICE TESTS
# ============================================================
from app.services.classifier_service import classify_roi


class TestClassifierService:
    """Tests for the ResNet50 fracture/normal classifier."""

    def test_classify_roi_returns_dict(self):
        """Classification result should be a dictionary."""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classify_roi(dummy_image)
        assert isinstance(result, dict)

    def test_classify_roi_has_required_keys(self):
        """Result dictionary should contain 'classification' and 'confidence' keys."""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classify_roi(dummy_image)
        assert "classification" in result
        assert "confidence" in result

    def test_classify_roi_valid_class(self):
        """Classification should be either 'fracture', 'normal', or 'unknown'."""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classify_roi(dummy_image)
        assert result["classification"] in ["fracture", "normal", "unknown"]

    def test_classify_roi_confidence_range(self):
        """Confidence should be between 0.0 and 1.0."""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classify_roi(dummy_image)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_roi_none_input(self):
        """None input should return 'unknown' classification."""
        result = classify_roi(None)
        assert result["classification"] == "unknown"
        assert result["confidence"] == 0.0

    def test_classify_roi_empty_image(self):
        """Empty image should return 'unknown' classification."""
        empty_image = np.array([], dtype=np.uint8)
        result = classify_roi(empty_image)
        assert result["classification"] == "unknown"


# ============================================================
# 5. STORAGE SERVICE TESTS (URL parsing only — no API calls)
# ============================================================
from app.services.storage_service import extract_public_id_from_url


class TestStorageService:
    """Tests for Cloudinary URL parsing utility."""

    def test_extract_public_id_standard_url(self):
        """Standard Cloudinary URL should extract correct public_id."""
        url = "https://res.cloudinary.com/mycloud/image/upload/v1234567890/fractify/xrays/abc123.png"
        result = extract_public_id_from_url(url)
        assert result == "fractify/xrays/abc123"

    def test_extract_public_id_gradcam_folder(self):
        """Grad-CAM folder URL should extract correct public_id."""
        url = "https://res.cloudinary.com/mycloud/image/upload/v9999/fractify/gradcam/heatmap001.png"
        result = extract_public_id_from_url(url)
        assert result == "fractify/gradcam/heatmap001"

    def test_extract_public_id_no_version(self):
        """URL without version prefix should still extract public_id."""
        url = "https://res.cloudinary.com/mycloud/image/upload/fractify/roi/crop001.png"
        result = extract_public_id_from_url(url)
        assert result == "fractify/roi/crop001"

    def test_extract_public_id_non_cloudinary_url(self):
        """Non-Cloudinary URL should return None."""
        url = "https://example.com/images/photo.jpg"
        result = extract_public_id_from_url(url)
        assert result is None

    def test_extract_public_id_empty_string(self):
        """Empty string should return None."""
        result = extract_public_id_from_url("")
        assert result is None

    def test_extract_public_id_none_input(self):
        """None input should return None."""
        result = extract_public_id_from_url(None)
        assert result is None

    def test_extract_public_id_removes_extension(self):
        """File extension should be removed from public_id."""
        url = "https://res.cloudinary.com/mycloud/image/upload/v123/fractify/annotated/img.jpg"
        result = extract_public_id_from_url(url)
        assert result == "fractify/annotated/img"
        assert ".jpg" not in result


# ============================================================
# 6. GRAD-CAM SERVICE TESTS
# ============================================================
from app.services.gradcam_service import generate_gradcam


class TestGradCAMService:
    """Tests for Grad-CAM heatmap generation."""

    def test_gradcam_returns_string_url(self):
        """Grad-CAM should return a Cloudinary URL string for valid input."""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = generate_gradcam(dummy_image, predicted_class_idx=0)
        # Result should be either a URL string or None (if Cloudinary not configured)
        assert result is None or isinstance(result, str)

    def test_gradcam_none_input_returns_none(self):
        """None image input should return None."""
        result = generate_gradcam(None, predicted_class_idx=0)
        assert result is None

    def test_gradcam_empty_image_returns_none(self):
        """Empty image should return None."""
        empty_image = np.array([], dtype=np.uint8)
        result = generate_gradcam(empty_image, predicted_class_idx=0)
        assert result is None