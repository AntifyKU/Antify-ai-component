"""
Antify AI Model Server
FastAPI server for BioCLIP ant species classification.
Runs on port 8001 and serves the fine-tuned BioCLIP model.
"""

from __future__ import annotations

import io
import os
import sys
import time
import threading
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Heavy ML imports (torch, open_clip, BioCLIPClassifier) are deferred to
# load_model() which runs in a background thread — uvicorn opens the port
# instantly and Cloud Run's startup probe passes within seconds.

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_preprocess = None
_tokenizer = None
_classes = None
_device = None
_model_load_time = None
_model_loading = False
_model_error = None

# Safety gate prompts (same as interactive_inference.py)
POSITIVE_PROMPTS = [
    "ant", "insect", "bug", "ant colony", "ants on a leaf",
    "macro photo of an ant", "specimen of an ant", "Hymenoptera",
    "winged ant", "ant queen", "hairy ant", "black ant",
    "ant specimen", "ant on white background", "pinned ant",
    "microscope photo of an ant", "yellow ant", "orange ant",
    "ant on a leaf", "nature photo of an ant", "wild ant",
]
NEGATIVE_PROMPTS = [
    "cat", "dog", "person", "human face",
    "cartoon", "drawing", "illustration", "clipart", "digital art", "vector graphics",
    "cartoon of an ant", "drawing of an ant", "ant illustration", "specimen illustration",
]

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bioclip_finetuned.pt"),
)


def _download_from_gcs():
    """Download model from GCS if MODEL_GCS_PATH is set and model doesn't exist locally."""
    gcs_path = os.environ.get("MODEL_GCS_PATH", "").strip()
    if not gcs_path or os.path.exists(MODEL_PATH):
        return
    if not gcs_path.startswith("gs://"):
        print(f"[Model Server] Invalid MODEL_GCS_PATH: {gcs_path}")
        return
    gcs_stripped = gcs_path[len("gs://"):]
    bucket_name, _, blob_name = gcs_stripped.partition("/")
    print(f"[Model Server] Downloading model from {gcs_path} ...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(MODEL_PATH)
    print(f"[Model Server] Model downloaded to {MODEL_PATH}")


def load_model():
    """Download (if needed) and load the BioCLIP model into memory (one-time)."""
    global _model, _preprocess, _tokenizer, _classes, _device, _model_load_time, _model_loading, _model_error

    try:
        _model_loading = True

        # Lazy imports — kept here so module-level load is fast and uvicorn
        # opens the port in <1s (Cloud Run's startup probe passes instantly)
        import torch
        import open_clip
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
        from bioclip_model import BioCLIPClassifier

        _download_from_gcs()

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model Server] Using device: {_device}")
        print(f"[Model Server] Loading model from {MODEL_PATH} ...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Set MODEL_GCS_PATH env var.")

        start = time.time()
        checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

        # Load class names
        if "classes" in checkpoint:
            _classes = checkpoint["classes"]
            print(f"[Model Server] Loaded {len(_classes)} classes from checkpoint.")
        else:
            raise ValueError("Checkpoint does not contain 'classes' key.")

        # Build model
        _model = BioCLIPClassifier(num_classes=len(_classes))
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model = _model.to(_device)
        _model.eval()

        # Transforms & tokenizer
        _, _preprocess = _model.get_transforms()
        _tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

        _model_load_time = time.time() - start
        print(f"[Model Server] Model loaded in {_model_load_time:.1f}s — ready to serve!")
    except Exception as e:
        _model_error = str(e)
        print(f"[Model Server] ERROR loading model: {e}")
    finally:
        _model_loading = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in a background thread so uvicorn opens the port immediately.
    # Cloud Run's health check will pass as soon as the port is open.
    # /classify returns 503 until the model is ready.
    t = threading.Thread(target=load_model, daemon=True)
    t.start()
    yield


app = FastAPI(
    title="Antify AI Model Server",
    description="BioCLIP ant species classification server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safety_check(image_tensor: torch.Tensor) -> dict | None:
    """
    Run the safety gate. Returns None if the image passes.
    Returns a dict with rejection details if it fails.
    """
    safety_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    safety_tokens = _tokenizer(safety_prompts).to(_device)

    with torch.no_grad():
        img_features = _model.backbone.encode_image(image_tensor)
        text_features = _model.backbone.encode_text(safety_tokens)

        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        safety_probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)

        positive_score = safety_probs[0][: len(POSITIVE_PROMPTS)].sum().item()
        negative_score = safety_probs[0][len(POSITIVE_PROMPTS) :].sum().item()

        top_idx = safety_probs[0].argmax().item()
        top_prob = safety_probs[0][top_idx].item()

        if negative_score > 0.6 and top_idx >= len(POSITIVE_PROMPTS):
            return {
                "rejected": True,
                "detected_as": safety_prompts[top_idx],
                "confidence": round(top_prob, 4),
                "ant_score": round(positive_score, 4),
                "non_ant_score": round(negative_score, 4),
            }

    return None


def _classify(image_tensor: torch.Tensor, top_k: int = 5) -> list[dict]:
    """Run classification and return list of predictions."""
    with torch.no_grad():
        logits = _model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_labels = probs.cpu().topk(min(top_k, len(_classes)), dim=1)

    predictions = []
    for i in range(top_probs.size(1)):
        label_idx = top_labels[0][i].item()
        predictions.append(
            {
                "rank": i + 1,
                "class_name": _classes[label_idx],
                "confidence": round(top_probs[0][i].item(), 6),
            }
        )
    return predictions


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_loading": _model_loading,
        "model_error": _model_error,
        "device": str(_device) if _device else None,
        "num_classes": len(_classes) if _classes else 0,
        "model_load_time_s": round(_model_load_time, 2) if _model_load_time else None,
    }


@app.get("/models")
async def list_models():
    return [
        {
            "name": "bioclip_finetuned",
            "type": "BioCLIP + Linear Head",
            "num_classes": len(_classes) if _classes else 0,
            "device": str(_device) if _device else None,
        }
    ]


@app.post("/classify")
async def classify(
    file: UploadFile = File(..., description="Image file to classify"),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Min confidence threshold"),
    top_k: int = Query(5, ge=1, le=20, description="Number of top predictions"),
):
    """
    Classify an ant image and return species predictions.
    Applies a safety gate first to reject non-ant images.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read and preprocess image
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Debug logging
        import hashlib
        img_hash = hashlib.md5(content).hexdigest()[:8]
        print(f"[classify] Image: {file.filename}, size={len(content)} bytes, hash={img_hash}, dimensions={image.size}")
        
        image_tensor = _preprocess(image).unsqueeze(0).to(_device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Safety gate
    rejection = _safety_check(image_tensor)
    if rejection:
        print(f"[classify] REJECTED: {rejection}")
        return {
            "success": False,
            "message": f"Image rejected: detected as '{rejection['detected_as']}' instead of an ant.",
            "safety": rejection,
            "top_prediction": "Not an ant",
            "top_confidence": 0.0,
            "top_predictions": [],
            "model": "bioclip_finetuned",
        }

    # Classification
    predictions = _classify(image_tensor, top_k=top_k)
    
    # Debug logging
    print(f"[classify] Results for hash={img_hash}:")
    for p in predictions:
        print(f"  #{p['rank']} {p['class_name']}: {p['confidence']*100:.2f}%")

    # Filter by confidence threshold
    filtered = [p for p in predictions if p["confidence"] >= confidence]

    top = predictions[0] if predictions else {"class_name": "Unknown", "confidence": 0.0}

    return {
        "success": True,
        "top_prediction": top["class_name"],
        "top_confidence": round(top["confidence"], 6),
        "top_predictions": predictions,  # return all, let client filter
        "model": "bioclip_finetuned",
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    print(f"[Model Server] Starting on port {port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
