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
from app.services.storage_service import upload_file_to_cloudinary
from app.services.storage_service import delete_from_cloudinary, extract_public_id_from_url

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

    # Upload to Cloudinary
    try:
        result = upload_file_to_cloudinary(contents, folder="fractify/xrays")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Image upload failed: {str(e)}")

    # Create database record
    xray_upload = XrayUpload(
        user_id=current_user.id,
        original_filename=file.filename,
        file_path=result["url"],   # now stores Cloudinary URL
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



@router.delete("/{upload_id}")
def delete_upload(upload_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    upload = db.query(XrayUpload).filter(
        XrayUpload.id == upload_id,
        XrayUpload.user_id == current_user.id
    ).first()

    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Collect all Cloudinary URLs to delete
    urls_to_delete = [upload.file_path, upload.annotated_path]
    for pred in upload.predictions:
        urls_to_delete.append(pred.roi_image_path)
        urls_to_delete.append(pred.gradcam_image_path)

    # Delete each asset from Cloudinary
    for url in urls_to_delete:
        if url:
            public_id = extract_public_id_from_url(url)
            if public_id:
                delete_from_cloudinary(public_id)

    # Delete DB records
    db.delete(upload)
    db.commit()
    return {"message": "Upload deleted successfully"}


@router.get("/", response_model=list[XrayUploadResponse])
def get_all_uploads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    uploads = db.query(XrayUpload).filter(
        XrayUpload.user_id == current_user.id
    ).order_by(XrayUpload.created_at.desc()).all()

    return [XrayUploadResponse.model_validate(u) for u in uploads]