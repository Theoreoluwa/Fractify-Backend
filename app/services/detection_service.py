from inference_sdk import InferenceHTTPClient
from app.config import settings
import math

# Cache these at module load so threads don't trigger config reads
_ROBOFLOW_API_URL = settings.ROBOFLOW_API_URL
_ROBOFLOW_API_KEY = settings.ROBOFLOW_API_KEY
_ROBOFLOW_MODEL_ID_ = settings.ROBOFLOW_MODEL_ID
_FRACTURE_MODEL_ID = settings.FRACTURE_MODEL_ID

def get_detections(image_path: str) -> list:
    """image_path is now a Cloudinary URL — Roboflow accepts URLs directly."""
    client = InferenceHTTPClient(
        api_url=settings.ROBOFLOW_API_URL,
        api_key=settings.ROBOFLOW_API_KEY
    )
    result = client.infer(image_path, model_id=settings.ROBOFLOW_MODEL_ID)
    predictions = result.get("predictions", [])
    return predictions


def get_fracture_detections(image_path: str) -> list:
    """image_path is now a Cloudinary URL — Roboflow accepts URLs directly."""
    client = InferenceHTTPClient(
        api_url=settings.ROBOFLOW_API_URL,
        api_key=settings.ROBOFLOW_API_KEY
    )
    result = client.infer(image_path, model_id=settings.FRACTURE_MODEL_ID)
    predictions = result.get("predictions", [])
    return predictions


def check_overlap(anatomy_box, fracture_box, threshold=0.2):
    a_x1 = anatomy_box["x"] - anatomy_box["width"] / 2
    a_y1 = anatomy_box["y"] - anatomy_box["height"] / 2
    a_x2 = anatomy_box["x"] + anatomy_box["width"] / 2
    a_y2 = anatomy_box["y"] + anatomy_box["height"] / 2

    f_x1 = fracture_box["x"] - fracture_box["width"] / 2
    f_y1 = fracture_box["y"] - fracture_box["height"] / 2
    f_x2 = fracture_box["x"] + fracture_box["width"] / 2
    f_y2 = fracture_box["y"] + fracture_box["height"] / 2

    inter_x1 = max(a_x1, f_x1)
    inter_y1 = max(a_y1, f_y1)
    inter_x2 = min(a_x2, f_x2)
    inter_y2 = min(a_y2, f_y2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False, 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    anatomy_area = anatomy_box["width"] * anatomy_box["height"]

    if anatomy_area == 0:
        return False, 0.0

    overlap_ratio = inter_area / anatomy_area
    return overlap_ratio >= threshold, overlap_ratio


def classify_regions_by_overlap(anatomy_predictions, fracture_predictions, confidence_threshold=0.5):
    results = []

    for anat in anatomy_predictions:
        if anat.get("confidence", 0) < confidence_threshold:
            continue

        is_fractured = False
        best_overlap = 0.0
        fracture_confidence = 0.0

        for frac in fracture_predictions:
            if frac.get("confidence", 0) < confidence_threshold:
                continue

            overlaps, overlap_ratio = check_overlap(anat, frac, threshold=0.1)
            if overlaps and overlap_ratio > best_overlap:
                is_fractured = True
                best_overlap = overlap_ratio
                fracture_confidence = frac.get("confidence", 0)

        results.append({
            "anatomy": anat,
            "classification": "fracture" if is_fractured else "normal",
            "confidence": fracture_confidence if is_fractured else 1.0 - best_overlap,
            "overlap_ratio": best_overlap
        })

    return results


def find_closest_region(fracture_box, anatomy_predictions, confidence_threshold=0.5):
    frac_cx = fracture_box["x"]
    frac_cy = fracture_box["y"]

    closest_region = None
    closest_distance = float("inf")

    for anat in anatomy_predictions:
        if anat.get("confidence", 0) < confidence_threshold:
            continue

        anat_cx = anat["x"]
        anat_cy = anat["y"]

        distance = math.sqrt((frac_cx - anat_cx) ** 2 + (frac_cy - anat_cy) ** 2)

        if distance < closest_distance:
            closest_distance = distance
            closest_region = anat.get("class", "unknown")

    return closest_region, closest_distance