import os
import uuid
import shutil
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.prediction import XrayUpload
from app.schemas.prediction import XrayUploadResponse
from app.middleware.auth_middleware import get_current_user
from app.config import settings
from app.models.prediction import PredictionResult, XrayUpload

router = APIRouter(prefix="/xray", tags=["X-ray Upload"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/upload", response_model=XrayUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_xray(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )

    # Validate image is not corrupted
    import cv2
    import numpy as np
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image or is corrupted"
        )

    # Generate unique filename and save
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    save_path = os.path.join(settings.UPLOAD_DIR, "xrays", unique_filename)

    with open(save_path, "wb") as buffer:
        buffer.write(contents)

    # Create database record
    xray_upload = XrayUpload(
        user_id=current_user.id,
        original_filename=file.filename,
        file_path=save_path,
        status="uploaded"
    )

    db.add(xray_upload)
    db.commit()
    db.refresh(xray_upload)

    return XrayUploadResponse.model_validate(xray_upload)


@router.get("/{upload_id}", response_model=XrayUploadResponse)
def get_upload(
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

    return XrayUploadResponse.model_validate(upload)

@router.get("/{upload_id}/status")
def get_upload_status(
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

    return {
        "id": upload.id,
        "status": upload.status,
        "overall_severity": upload.overall_severity
    }

@router.delete("/{upload_id}", status_code=status.HTTP_200_OK)
def delete_upload(
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

    if upload.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete while processing"
        )

    # Get predictions first before deleting anything
    predictions = db.query(PredictionResult).filter(
        PredictionResult.upload_id == upload_id
    ).all()

    # Delete ROI and gradcam files
    for pred in predictions:
        for path in [pred.roi_image_path, pred.gradcam_image_path]:
            if path and os.path.exists(path):
                os.remove(path)

    # Delete prediction records
    db.query(PredictionResult).filter(
        PredictionResult.upload_id == upload_id
    ).delete()

    # Delete upload files
    for path in [upload.file_path, upload.annotated_path]:
        if path and os.path.exists(path):
            os.remove(path)

    # Delete upload record
    db.delete(upload)
    db.commit()

    return {"message": "Upload and all associated data deleted successfully"}


@router.get("/", response_model=list[XrayUploadResponse])
def get_all_uploads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    uploads = db.query(XrayUpload).filter(
        XrayUpload.user_id == current_user.id
    ).order_by(XrayUpload.created_at.desc()).all()

    return [XrayUploadResponse.model_validate(u) for u in uploads]