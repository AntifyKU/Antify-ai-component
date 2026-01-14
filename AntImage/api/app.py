"""
FastAPI REST API for Ant Detection/Classification using YOLO.

Run with:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

API Endpoints:
    POST /detect       - Detect ants in an image (bounding boxes)
    POST /classify     - Classify ant species in an image
    GET  /health       - Health check
    GET  /models       - List available models
"""

import io
import os
import base64
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ============================================================================
# Configuration
# ============================================================================

# Default model paths - adjust these to your model locations
DEFAULT_DETECT_MODEL = "models/ant_detector_v1.pt"
DEFAULT_CLASSIFY_MODEL = None  # Set to classification model path if available

# Global model cache
models = {}


# ============================================================================
# Pydantic Models for API responses
# ============================================================================

class Detection(BaseModel):
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    """Response for detection endpoint."""
    success: bool
    message: str
    num_detections: int
    detections: list[Detection]
    image_size: list[int]  # [width, height]


class ClassificationResult(BaseModel):
    """Single classification result."""
    rank: int
    class_name: str
    confidence: float


class ClassificationResponse(BaseModel):
    """Response for classification endpoint."""
    success: bool
    message: str
    top_prediction: str
    top_confidence: float
    top5_predictions: list[ClassificationResult]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: list[str]


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    path: str
    type: str
    loaded: bool


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str, model_name: str) -> Optional[YOLO]:
    """Load a YOLO model and cache it."""
    if model_name in models:
        return models[model_name]
    
    if not os.path.exists(model_path):
        return None
    
    try:
        model = YOLO(model_path)
        models[model_name] = model
        print(f"✅ Loaded model: {model_name} from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("🚀 Starting Ant Detection API...")
    
    # Load detection model
    if DEFAULT_DETECT_MODEL and os.path.exists(DEFAULT_DETECT_MODEL):
        load_model(DEFAULT_DETECT_MODEL, "detect")
    
    # Load classification model if available
    if DEFAULT_CLASSIFY_MODEL and os.path.exists(DEFAULT_CLASSIFY_MODEL):
        load_model(DEFAULT_CLASSIFY_MODEL, "classify")
    
    print(f"📦 Models loaded: {list(models.keys())}")
    yield
    
    # Cleanup
    models.clear()
    print("👋 API shutdown complete")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Ant Detection API",
    description="REST API for detecting and classifying ants using YOLO models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

async def read_image(file: UploadFile) -> np.ndarray:
    """Read uploaded image file to numpy array."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return image


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def draw_detections(image: np.ndarray, results) -> np.ndarray:
    """Draw detection boxes on image."""
    annotated = image.copy()
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return annotated


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Ant Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /detect": "Detect ants in an image",
            "POST /classify": "Classify ant species",
            "GET /health": "Health check",
            "GET /models": "List available models"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys())
    )


@app.get("/models")
async def list_models():
    """List available models."""
    model_list = []
    
    # Check detection model
    model_list.append(ModelInfo(
        name="detect",
        path=DEFAULT_DETECT_MODEL or "Not configured",
        type="detection",
        loaded="detect" in models
    ))
    
    # Check classification model
    model_list.append(ModelInfo(
        name="classify",
        path=DEFAULT_CLASSIFY_MODEL or "Not configured",
        type="classification",
        loaded="classify" in models
    ))
    
    return {"models": model_list}


@app.post("/detect", response_model=DetectionResponse)
async def detect_ants(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
):
    """
    Detect ants in an uploaded image.
    
    Returns bounding boxes with class names and confidence scores.
    """
    # Check if model is loaded
    if "detect" not in models:
        raise HTTPException(
            status_code=503, 
            detail="Detection model not loaded. Check model path configuration."
        )
    
    # Read image
    image = await read_image(file)
    h, w = image.shape[:2]
    
    # Run inference
    model = models["detect"]
    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        verbose=False
    )
    
    # Parse results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append(Detection(
                class_id=int(box.cls[0]),
                class_name=result.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=box.xyxy[0].tolist()
            ))
    
    return DetectionResponse(
        success=True,
        message=f"Found {len(detections)} ant(s)",
        num_detections=len(detections),
        detections=detections,
        image_size=[w, h]
    )


@app.post("/detect/visualize")
async def detect_and_visualize(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS"),
    return_base64: bool = Query(False, description="Return base64 instead of image file")
):
    """
    Detect ants and return annotated image with bounding boxes drawn.
    """
    if "detect" not in models:
        raise HTTPException(status_code=503, detail="Detection model not loaded")
    
    image = await read_image(file)
    
    model = models["detect"]
    results = model.predict(source=image, conf=confidence, iou=iou, verbose=False)
    
    # Draw detections on image
    annotated = draw_detections(image, results)
    
    if return_base64:
        return {
            "success": True,
            "image_base64": encode_image_base64(annotated),
            "num_detections": sum(len(r.boxes) for r in results)
        }
    
    # Return as image file
    _, buffer = cv2.imencode('.jpg', annotated)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=detection_result.jpg"}
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_ant(
    file: UploadFile = File(..., description="Image file to classify"),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold")
):
    """
    Classify ant species in an uploaded image.
    
    Returns top prediction and top-5 predictions with confidence scores.
    """
    if "classify" not in models:
        raise HTTPException(
            status_code=503,
            detail="Classification model not loaded. Check model path configuration."
        )
    
    image = await read_image(file)
    
    model = models["classify"]
    results = model.predict(source=image, conf=confidence, verbose=False)
    
    if not results or results[0].probs is None:
        raise HTTPException(status_code=500, detail="Classification failed")
    
    probs = results[0].probs
    names = results[0].names
    
    # Get top prediction
    top1_idx = probs.top1
    top1_conf = float(probs.top1conf)
    top1_name = names[top1_idx]
    
    # Get top 5 predictions
    top5_results = []
    for rank, (idx, conf) in enumerate(zip(probs.top5, probs.top5conf), 1):
        top5_results.append(ClassificationResult(
            rank=rank,
            class_name=names[idx],
            confidence=float(conf)
        ))
    
    return ClassificationResponse(
        success=True,
        message=f"Classification complete",
        top_prediction=top1_name,
        top_confidence=top1_conf,
        top5_predictions=top5_results
    )


@app.post("/predict")
async def predict_auto(
    file: UploadFile = File(..., description="Image file to analyze"),
    mode: str = Query("detect", description="Prediction mode: 'detect' or 'classify'"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold")
):
    """
    Auto-detect prediction mode and run inference.
    
    This is a convenience endpoint that routes to /detect or /classify based on mode.
    """
    if mode == "detect":
        return await detect_ants(file, confidence)
    elif mode == "classify":
        return await classify_ant(file, confidence)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'detect' or 'classify'")


# ============================================================================
# Run with: uvicorn api.app:app --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
