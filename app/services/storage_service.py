import os
import uuid
import cloudinary
import cloudinary.uploader
import numpy as np
import cv2
from typing import Optional
from app.config import settings


# Configure Cloudinary once at module load
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)


def upload_file_to_cloudinary(file_bytes: bytes, folder: str = "fractify/xrays") -> dict:
    """
    Upload raw file bytes (e.g. from a FastAPI UploadFile) to Cloudinary.
    Returns a dict with 'url' (public URL) and 'public_id' (for later deletion).
    """
    public_id = f"{folder}/{uuid.uuid4().hex}"
    result = cloudinary.uploader.upload(
        file_bytes,
        public_id=public_id,
        folder=folder,
        resource_type="image",
        overwrite=False,
    )
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
    }


def upload_numpy_image(image: np.ndarray, folder: str = "fractify/generated") -> dict:
    """
    Upload an OpenCV/numpy image (BGR format) to Cloudinary.
    Used for annotated images, ROI crops, and Grad-CAM heatmaps.
    """
    # Encode numpy array to PNG bytes
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")

    return upload_file_to_cloudinary(buffer.tobytes(), folder=folder)


def delete_from_cloudinary(public_id: str) -> bool:
    """
    Delete an asset from Cloudinary by its public_id.
    Returns True on success.
    """
    try:
        result = cloudinary.uploader.destroy(public_id, resource_type="image")
        return result.get("result") == "ok"
    except Exception:
        return False


def extract_public_id_from_url(url: str) -> Optional[str]:
    """
    Extract the Cloudinary public_id from a secure_url.
    Example:
        https://res.cloudinary.com/<cloud>/image/upload/v1234/fractify/xrays/abc.png
        -> fractify/xrays/abc
    """
    if not url or "cloudinary.com" not in url:
        return None
    try:
        # Split at '/upload/' and grab everything after
        after_upload = url.split("/upload/", 1)[1]
        # Remove version prefix (v1234/)
        parts = after_upload.split("/", 1)
        if parts[0].startswith("v") and parts[0][1:].isdigit():
            path = parts[1]
        else:
            path = after_upload
        # Remove file extension
        return os.path.splitext(path)[0]
    except (IndexError, AttributeError):
        return None