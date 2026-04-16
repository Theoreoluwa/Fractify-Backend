import os
import uuid
import cv2
import numpy as np
import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor

from app.database import get_db
from app.models.user import User
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
from app.config import settings

router = APIRouter(prefix="/predict", tags=["Prediction Pipeline"])

CONFIDENCE_THRESHOLD = 0.6


@router.post("/{upload_id}", response_model=XrayUploadResponse)
def run_pipeline(
    upload_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
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

    # ✅ LOAD IMAGE FROM CLOUDINARY (ONCE)
    try:
        response = requests.get(upload.file_path, timeout=30)
        response.raise_for_status()

        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Image decoding failed")

    except Exception as e:
        upload.status = "failed"
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load image: {str(e)}"
        )

    upload.status = "processing"
    db.commit()

    try:
        # ✅ RUN DETECTIONS ON IMAGE (NOT PATH)
        def run_anatomy():
            return get_detections(image)

        def run_fracture():
            return get_fracture_detections(image)

        with ThreadPoolExecutor(max_workers=2) as executor:
            anatomy_future = executor.submit(run_anatomy)
            fracture_future = executor.submit(run_fracture)

            anatomy_detections = anatomy_future.result(timeout=30)

            try:
                fracture_detections = fracture_future.result(timeout=30)
            except Exception:
                fracture_detections = []

        # Step 3: Overlap classification
        region_classifications = classify_regions_by_overlap(
            anatomy_detections, fracture_detections, CONFIDENCE_THRESHOLD
        )

        # Step 3b: unmatched fractures
        unmatched_fractures = []
        for frac in fracture_detections:
            if frac.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                continue

            matched = False
            for anat in anatomy_detections:
                if anat.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                    continue

                overlaps, _ = check_overlap(anat, frac, threshold=0.1)
                if overlaps:
                    matched = True
                    break

            if not matched:
                unmatched_fractures.append(frac)

        # ✅ PASS IMAGE (NOT PATH)
        crops = crop_all_regions(image, anatomy_detections)

        if not crops and not unmatched_fractures:
            upload.status = "completed"
            upload.overall_severity = "NONE"
            db.commit()
            db.refresh(upload)
            return XrayUploadResponse.model_validate(upload)

        original_image = image.copy()
        annotated_image = original_image.copy()

        all_results = []

        # Step 6: Process crops
        for crop_data in crops:
            cropped_image = crop_data["image"]
            pred = crop_data["prediction"]

            confidence = pred.get("confidence", 0)
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            region_class = pred.get("class", "unknown")

            classification = "normal"
            classification_confidence = 0.5

            for rc in region_classifications:
                if rc["anatomy"] == pred:
                    classification = rc["classification"]
                    classification_confidence = rc["confidence"]
                    break

            # GradCAM
            gradcam_path = None
            if classification == "fracture":
                try:
                    gradcam_path = generate_gradcam(cropped_image, 0)
                except Exception:
                    pass

            # Upload ROI
            try:
                roi_result = upload_numpy_image(cropped_image, folder="fractify/roi")
                roi_path = roi_result["url"]
            except Exception:
                roi_path = None

            severity = get_region_severity(
                region_class,
                classification,
                classification_confidence
            )

            # Draw box
            cx, cy = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])

            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)

            color = (0, 0, 255) if classification == "fracture" else (0, 255, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            db.add(PredictionResult(
                upload_id=upload.id,
                region_class=region_class,
                bbox_x=pred["x"],
                bbox_y=pred["y"],
                bbox_width=pred["width"],
                bbox_height=pred["height"],
                detection_confidence=confidence,
                classification=classification,
                classification_confidence=classification_confidence,
                roi_image_path=roi_path,
                gradcam_image_path=gradcam_path,
                severity=severity
            ))

            all_results.append({"classification": classification, "severity": severity})

        # Step 6b: unmatched fractures
        for frac in unmatched_fractures:
            cropped_frac = crop_region(original_image, frac, padding=40)

            if cropped_frac is None:
                continue

            try:
                roi_result = upload_numpy_image(cropped_frac, folder="fractify/roi")
                roi_path = roi_result["url"]
            except Exception:
                roi_path = None

            severity = get_region_severity("unknown", "fracture", frac.get("confidence", 0))

            db.add(PredictionResult(
                upload_id=upload.id,
                region_class="Bone Fracture",
                bbox_x=frac["x"],
                bbox_y=frac["y"],
                bbox_width=frac["width"],
                bbox_height=frac["height"],
                detection_confidence=frac["confidence"],
                classification="fracture",
                classification_confidence=frac["confidence"],
                roi_image_path=roi_path,
                gradcam_image_path=None,
                severity=severity
            ))

            all_results.append({"classification": "fracture", "severity": severity})

        # Annotated image upload
        try:
            annotated_result = upload_numpy_image(annotated_image, folder="fractify/annotated")
            annotated_path = annotated_result["url"]
        except Exception:
            annotated_path = upload.file_path

        upload.annotated_path = annotated_path
        upload.overall_severity = get_overall_severity(all_results)
        upload.status = "completed"

        db.commit()
        db.refresh(upload)

        return XrayUploadResponse.model_validate(upload)

    except Exception as e:
        upload.status = "failed"
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )