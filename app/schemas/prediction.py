from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class PredictionResultResponse(BaseModel):
    id: int
    region_class: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    detection_confidence: float
    classification: Optional[str] = None
    classification_confidence: Optional[float] = None
    roi_image_path: Optional[str] = None
    gradcam_image_path: Optional[str] = None
    severity: Optional[str] = None

    class Config:
        from_attributes = True


class XrayUploadResponse(BaseModel):
    id: int
    original_filename: str
    file_path: str
    annotated_path: Optional[str] = None
    overall_severity: Optional[str] = None
    status: str
    created_at: datetime
    predictions: List[PredictionResultResponse] = []

    class Config:
        from_attributes = True