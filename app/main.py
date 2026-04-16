import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.database import engine, Base
from app.config import settings
from app.models import User, XrayUpload, PredictionResult
from app.routes import auth
from app.routes import xray, prediction

app = FastAPI(
    title="Fracture Triage System API",
    description="AI-powered fracture detection and triage for hand and wrist radiographs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://fractify.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

os.makedirs(os.path.join(settings.UPLOAD_DIR, "xrays"), exist_ok=True)
os.makedirs(os.path.join(settings.UPLOAD_DIR, "gradcam"), exist_ok=True)
os.makedirs(os.path.join(settings.UPLOAD_DIR, "reports"), exist_ok=True)

app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.include_router(auth.router, prefix="/api")
app.include_router(xray.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Fracture Triage System API",
        "status": "running",
        "docs": "Visit /docs for interactive API documentation"
    }