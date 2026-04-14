import cv2
import numpy as np


def crop_region(image: np.ndarray, prediction: dict, padding: int = 40) -> np.ndarray:
    img_height, img_width = image.shape[:2]

    cx = prediction["x"]
    cy = prediction["y"]
    w = prediction["width"]
    h = prediction["height"]

    x1 = max(0, int(cx - w / 2) - padding)
    y1 = max(0, int(cy - h / 2) - padding)
    x2 = min(img_width, int(cx + w / 2) + padding)
    y2 = min(img_height, int(cy + h / 2) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def crop_all_regions(image_path: str, predictions: list) -> list:
    image = cv2.imread(image_path)
    if image is None:
        return []

    crops = []
    for pred in predictions:
        cropped = crop_region(image, pred)
        if cropped is not None and cropped.size > 0:
            crops.append({
                "image": cropped,
                "prediction": pred
            })

    return crops