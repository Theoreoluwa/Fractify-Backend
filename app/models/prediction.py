from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class XrayUpload(Base):
    __tablename__ = "xray_uploads"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    annotated_path = Column(String(500), nullable=True)
    overall_severity = Column(String(50), nullable=True)
    status = Column(String(50), default="uploaded")
    created_at = Column(DateTime, server_default=func.now())

    predictions = relationship("PredictionResult", back_populates="xray_upload")


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    upload_id = Column(Integer, ForeignKey("xray_uploads.id"), nullable=False)

    region_class = Column(String(50), nullable=False)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    detection_confidence = Column(Float, nullable=False)

    classification = Column(String(50), nullable=True)
    classification_confidence = Column(Float, nullable=True)

    roi_image_path = Column(String(500), nullable=True)
    gradcam_image_path = Column(String(500), nullable=True)

    severity = Column(String(50), nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    xray_upload = relationship("XrayUpload", back_populates="predictions")