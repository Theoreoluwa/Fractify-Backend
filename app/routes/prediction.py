import os
import uuid
import cv2
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.prediction import XrayUpload, PredictionResult
from app.schemas.prediction import XrayUploadResponse
from app.middleware.auth_middleware import get_current_user
from app.services.detection_service import get_detections, get_fracture_detections, classify_regions_by_overlap, check_overlap, find_closest_region
from app.services.roi_service import crop_all_regions
from app.services.classifier_service import classify_roi
from app.services.gradcam_service import generate_gradcam
from app.services.triage_service import get_region_severity, get_overall_severity
from app.config import settings
from concurrent.futures import ThreadPoolExecutor
import requests
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found"
        )

    if upload.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This upload has already been processed"
        )

    if upload.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This upload is currently being processed"
        )

    if not os.path.exists(upload.file_path):
        upload.status = "failed"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file not found on server"
        )

    upload.status = "processing"
    db.commit()

    try:
        # Steps 1 & 2: Run both detectors in PARALLEL on the full image
        anatomy_detections = []
        fracture_detections = []

        def run_anatomy():
            return get_detections(upload.file_path)

        def run_fracture():
            return get_fracture_detections(upload.file_path)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                anatomy_future = executor.submit(run_anatomy)
                fracture_future = executor.submit(run_fracture)

                anatomy_detections = anatomy_future.result(timeout=30)
                try:
                    fracture_detections = fracture_future.result(timeout=30)
                except Exception:
                    fracture_detections = []
        except Exception as e:
            upload.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Detection service unavailable: {str(e)}"
            )

        # Step 3: Classify regions by bounding box overlap
        region_classifications = classify_regions_by_overlap(
            anatomy_detections, fracture_detections, CONFIDENCE_THRESHOLD
        )

        # Step 3b: Find fractures that don't overlap with ANY anatomical region
        unmatched_fractures = []
        for frac in fracture_detections:
            if frac.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                continue
            matched = False
            for anat in anatomy_detections:
                if anat.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                    continue
                from app.services.detection_service import check_overlap
                overlaps, _ = check_overlap(anat, frac, threshold=0.1)
                if overlaps:
                    matched = True
                    break
            if not matched:
                unmatched_fractures.append(frac)

        # Step 4: Crop all detected regions
        crops = crop_all_regions(upload.file_path, anatomy_detections)

        if not crops and not unmatched_fractures:
            upload.status = "completed"
            upload.overall_severity = "NONE"
            db.commit()
            db.refresh(upload)
            return XrayUploadResponse.model_validate(upload)

        # Step 5: Draw bounding boxes on original image
        # NEW — fetch image bytes from Cloudinary URL
        try:
            response = requests.get(upload.file_path, timeout=30)
            response.raise_for_status()
            nparr = np.frombuffer(response.content, np.uint8)
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            upload.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch uploaded image: {str(e)}"
            )
        if original_image is None:
            upload.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to read uploaded image for annotation"
            )

        annotated_image = original_image.copy()

        # Step 6: Process each anatomical region
        all_results = []

        for crop_data in crops:
            cropped_image = crop_data["image"]
            pred = crop_data["prediction"]

            confidence = pred.get("confidence", 0)
            region_class = pred.get("class", "unknown")

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Find the classification from overlap analysis
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

            # Generate Grad-CAM heatmap ONLY for fracture regions
            gradcam_path = None
            if classification == "fracture":
                try:
                    class_names_list = ["fracture", "normal"]
                    predicted_idx = class_names_list.index(classification)
                    gradcam_path = generate_gradcam(cropped_image, predicted_idx)
                except Exception:
                    gradcam_path = None

            # Save cropped ROI image
            try:
                roi_result = upload_numpy_image(cropped_image, folder="fractify/roi")
                roi_path = roi_result["url"]
            except Exception:
                roi_path = None

            # Get triage severity
            severity = get_region_severity(
                region_class,
                classification,
                classification_confidence
            )

            # Draw bounding box
            cx, cy = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            if classification == "fracture":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

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

        # Step 6b: Process unmatched fractures (fractures outside anatomical regions)
        for frac in unmatched_fractures:
            frac_confidence = frac.get("confidence", 0)

            # Find closest anatomical region
            closest_region, distance = find_closest_region(frac, anatomy_detections, CONFIDENCE_THRESHOLD)

            # Name the fracture based on closest region
            if closest_region:
                frac_region = f"Fractured {closest_region}"
            else:
                frac_region = "Bone Fracture"

            # Get severity based on closest anatomical region
            severity = get_region_severity(
                closest_region or "unknown",
                "fracture",
                frac_confidence
            )

            # Crop the fracture region from original image
            from app.services.roi_service import crop_region
            cropped_frac = crop_region(original_image, frac, padding=40)

            gradcam_path = None
            roi_path = None

            if cropped_frac is not None and cropped_frac.size > 0:
                try:
                    gradcam_path = generate_gradcam(cropped_frac, 0)
                except Exception:
                    gradcam_path = None

                roi_filename = f"{uuid.uuid4().hex}_roi.png"
                roi_path = os.path.join(settings.UPLOAD_DIR, "xrays", roi_filename)
                cv2.imwrite(roi_path, cropped_frac)

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

        # Step 7: Save annotated image
        try:
            annotated_result = upload_numpy_image(annotated_image, folder="fractify/annotated")
            annotated_path = annotated_result["url"]
        except Exception:
            annotated_path = upload.file_path  # fallback to original

        # Step 8: Calculate overall severity
        overall_severity = get_overall_severity(all_results)

        # Step 9: Update upload record
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}"
        )