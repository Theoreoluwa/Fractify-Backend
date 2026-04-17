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

    # =========================
    # 1. LOAD IMAGE FROM CLOUD
    # =========================
    try:
        response = requests.get(upload.file_path, timeout=30)
        response.raise_for_status()
        nparr = np.frombuffer(response.content, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_image is None:
            raise ValueError("cv2 failed to decode image")
    except Exception as e:
        upload.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Image load failed: {str(e)}")

    upload.status = "processing"
    db.commit()

    try:
        # =========================
        # 2. RUN DETECTION MODELS (parallel, passing URL directly)
        # =========================
        # Read values BEFORE entering threads to avoid SQLAlchemy session conflicts
        image_url = upload.file_path
        
        def run_anatomy():
            return get_detections(image_url)

        def run_fracture():
            return get_fracture_detections(image_url)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                anatomy_future = executor.submit(run_anatomy)
                fracture_future = executor.submit(run_fracture)

                anatomy_detections = anatomy_future.result(timeout=30)
                try:
                    fracture_detections = fracture_future.result(timeout=30)
                except Exception as e:
                    fracture_detections = []
                    print(f"[FRACTURE DETECTION FAILED]: {str(e)}")
                    print(f"[PIPELINE DEBUG] Anatomy: {len(anatomy_detections)}, Fractures: {len(fracture_detections)}")
                    print(f"[PIPELINE DEBUG] FRACTURE_MODEL_ID = '{settings.FRACTURE_MODEL_ID}'")
        except Exception as e:
            upload.status = "failed"
            db.commit()
            raise HTTPException(status_code=503, detail=f"Detection service unavailable: {str(e)}")

        if not anatomy_detections:
            upload.status = "completed"
            upload.overall_severity = "NONE"
            db.commit()
            db.refresh(upload)
            return XrayUploadResponse.model_validate(upload)

        # =========================
        # 3. CLASSIFY BY OVERLAP
        # =========================
        region_classifications = classify_regions_by_overlap(
            anatomy_detections, fracture_detections, CONFIDENCE_THRESHOLD
        )

        # 3b. Find unmatched fractures
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

        # =========================
        # 4. CROP REGIONS (fetches image from URL internally)
        # =========================
        crops = crop_all_regions(upload.file_path, anatomy_detections)

        if not crops and not unmatched_fractures:
            upload.status = "completed"
            upload.overall_severity = "NONE"
            db.commit()
            db.refresh(upload)
            return XrayUploadResponse.model_validate(upload)

        # =========================
        # 5. PREPARE ANNOTATED IMAGE
        # =========================
        annotated_image = original_image.copy()

        # =========================
        # 6. PROCESS EACH ANATOMICAL REGION
        # =========================
        all_results = []

        for crop_data in crops:
            cropped_image = crop_data["image"]
            pred = crop_data["prediction"]

            confidence = pred.get("confidence", 0)
            region_class = pred.get("class", "unknown")

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Find classification from overlap analysis
            classification_result = None
            for rc in region_classifications:
                if rc["anatomy"] == pred:
                    classification_result = rc
                    break

            if classification_result:
                classification = classification_result["classification"]
                classification_confidence = classification_result["confidence"]
            else:
                classification = "normal"
                classification_confidence = 0.5

            # Grad-CAM only for fractures
            gradcam_path = None
            if classification == "fracture":
                try:
                    class_names_list = ["fracture", "normal"]
                    predicted_idx = class_names_list.index(classification)
                    gradcam_path = generate_gradcam(cropped_image, predicted_idx)
                except Exception:
                    gradcam_path = None

            # Upload ROI crop to Cloudinary
            roi_path = None
            try:
                roi_result = upload_numpy_image(cropped_image, folder="fractify/roi")
                roi_path = roi_result["url"]
            except Exception:
                roi_path = None

            # Get triage severity
            severity = get_region_severity(region_class, classification, classification_confidence)

            # Draw bounding box on annotated image
            cx, cy = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            color = (0, 0, 255) if classification == "fracture" else (0, 255, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Save prediction result
            result = PredictionResult(
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
            )
            db.add(result)

            all_results.append({
                "classification": classification,
                "severity": severity
            })

        # =========================
        # 6b. PROCESS UNMATCHED FRACTURES
        # =========================
        for frac in unmatched_fractures:
            frac_confidence = frac.get("confidence", 0)

            closest_region, distance = find_closest_region(frac, anatomy_detections, CONFIDENCE_THRESHOLD)

            if closest_region:
                frac_region = f"Fractured {closest_region}"
            else:
                frac_region = "Bone Fracture"

            severity = get_region_severity(
                closest_region or "unknown",
                "fracture",
                frac_confidence
            )

            cropped_frac = crop_region(original_image, frac, padding=40)

            gradcam_path = None
            roi_path = None

            if cropped_frac is not None and cropped_frac.size > 0:
                try:
                    gradcam_path = generate_gradcam(cropped_frac, 0)
                except Exception:
                    gradcam_path = None

                try:
                    roi_result = upload_numpy_image(cropped_frac, folder="fractify/roi")
                    roi_path = roi_result["url"]
                except Exception:
                    roi_path = None

            # Draw red bounding box
            cx, cy = int(frac["x"]), int(frac["y"])
            w, h = int(frac["width"]), int(frac["height"])
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            result = PredictionResult(
                upload_id=upload.id,
                region_class=frac_region,
                bbox_x=frac["x"],
                bbox_y=frac["y"],
                bbox_width=frac["width"],
                bbox_height=frac["height"],
                detection_confidence=frac_confidence,
                classification="fracture",
                classification_confidence=frac_confidence,
                roi_image_path=roi_path,
                gradcam_image_path=gradcam_path,
                severity=severity
            )
            db.add(result)

            all_results.append({
                "classification": "fracture",
                "severity": severity
            })

        # =========================
        # 7. UPLOAD ANNOTATED IMAGE TO CLOUDINARY
        # =========================
        try:
            annotated_result = upload_numpy_image(annotated_image, folder="fractify/annotated")
            annotated_path = annotated_result["url"]
        except Exception:
            annotated_path = upload.file_path  # fallback to original

        # =========================
        # 8. CALCULATE OVERALL SEVERITY
        # =========================
        overall_severity = get_overall_severity(all_results)

        # =========================
        # 9. UPDATE UPLOAD RECORD
        # =========================
        upload.annotated_path = annotated_path
        upload.overall_severity = overall_severity
        upload.status = "completed"
        db.commit()
        db.refresh(upload)

        return XrayUploadResponse.model_validate(upload)

    except HTTPException:
        raise
    except Exception as e:
        upload.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")