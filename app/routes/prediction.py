import cv2
import numpy as np
import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor

from app.database import get_db
from app.models.prediction import XrayUpload, PredictionResult
from app.schemas.prediction import XrayUploadResponse
from app.middleware.auth_middleware import get_current_user

from app.services.detection_service import (
    get_detections,
    get_fracture_detections,
    classify_regions_by_overlap,
    find_closest_region,
    check_overlap
)

from app.services.roi_service import crop_all_regions, crop_region
from app.services.gradcam_service import generate_gradcam
from app.services.triage_service import get_region_severity, get_overall_severity
from app.services.storage_service import upload_numpy_image

router = APIRouter(prefix="/predict", tags=["Prediction Pipeline"])

CONFIDENCE_THRESHOLD = 0.6


@router.post("/{upload_id}", response_model=XrayUploadResponse)
def run_pipeline(
    upload_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):

    upload = db.query(XrayUpload).filter(
        XrayUpload.id == upload_id,
        XrayUpload.user_id == current_user.id
    ).first()

    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    if upload.status == "completed":
        raise HTTPException(status_code=400, detail="Already processed")

    if upload.status == "processing":
        raise HTTPException(status_code=409, detail="Already processing")

    # =========================
    # 1. LOAD IMAGE FROM CLOUD
    # =========================
    try:
        response = requests.get(upload.file_path, timeout=30)
        response.raise_for_status()

        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("cv2 failed to decode image")

    except Exception as e:
        upload.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Image load failed: {str(e)}")

    upload.status = "processing"
    db.commit()

    # =========================
    # 2. RUN DETECTION MODELS
    # =========================
    try:

        def run_anatomy():
            return get_detections(upload.file_path)

        def run_fracture():
            return get_fracture_detections(upload.file_path)

        with ThreadPoolExecutor(max_workers=2) as executor:
            anatomy_future = executor.submit(run_anatomy)
            fracture_future = executor.submit(run_fracture)

            anatomy_detections = anatomy_future.result(timeout=30)
            fracture_detections = fracture_future.result(timeout=30)

    except Exception as e:
        upload