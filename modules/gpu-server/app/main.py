"""
GPU Server - Facial Embedding & Emotion Analysis Service

Single Docker container serving multiple GPU-accelerated functions.
Public API uses the base path /iot/(endpoint) only (no /v1).
vLLM and other application backends run internally inside Docker.

1. POST /iot/vectorizer - Generate face embeddings (DeepFace/ArcFace)
2. POST /iot/emotions - Analyze emotions (VLM, internally via vLLM)

Endpoints defined in api_routes.yaml (base /iot). Health: GET /iot/health, GET /iot/models.
"""

import os
import time
import uuid
import base64
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, List

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from app.processors.embedding import EmbeddingProcessor
from app.processors.emotion import EmotionProcessor
from app.utils.logging import LogHandler


# ============================================
# Configuration
# ============================================

def load_config():
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    # Default config
    return {
        "server": {"host": "0.0.0.0", "port": 5000},
        "models": {
            "embedding": {"model_name": "ArcFace", "embedding_dim": 512},
            "emotion": {"backend": "mock"}
        },
        "logging": {"level": "INFO", "max_log_entries": 1000}
    }


config = load_config()


def load_routes():
    """Load API route paths from api_routes.yaml (or defaults if missing)."""
    routes_path = os.environ.get("ROUTES_CONFIG_PATH", "api_routes.yaml")
    if os.path.exists(routes_path):
        with open(routes_path, "r") as f:
            raw = yaml.safe_load(f)
    else:
        # Sensible defaults if file is missing
        raw = {
            "base": "/iot",
            "endpoints": {
                "health": "/health", "models": "/models",
                "vectorizer": "/vectorizer", "emotions": "/emotions",
                "transcription": "/transcription",
            }
        }
    base = raw.get("base", "/iot")
    return {k: f"{base}{v}" for k, v in raw.get("endpoints", {}).items()}


routes = load_routes()

# In-memory log storage
log_storage = deque(maxlen=config.get("logging", {}).get("max_log_entries", 1000))
websocket_connections = []

# Add custom handler to logger
log_handler = LogHandler(log_queue=log_storage, websocket_connections=websocket_connections)
logger.add(
    log_handler.write,
    format=config.get("logging", {}).get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
)

# Processors (initialized in lifespan)
embedding_processor: Optional[EmbeddingProcessor] = None
emotion_processor: Optional[EmotionProcessor] = None


# ============================================
# Request/Response Models - Vectorizer
# ============================================

class VectorizerImageInput(BaseModel):
    """Single image input for vectorizer"""
    pose: Optional[str] = None  # front, left, right, up, down
    data: str  # Base64 encoded image


class VectorizerRequest(BaseModel):
    """Request for POST /iot/vectorizer"""
    employee_id: str = Field(..., description="Employee/Person identifier")
    images: List[VectorizerImageInput] = Field(..., description="List of pose images to process")
    options: Optional[dict] = Field(default=None, description="Processing options")


class VectorizerResponse(BaseModel):
    """Response from POST /iot/vectorizer"""
    employee_id: str
    enrollmentProcessedFile: str  # Base64 Float32Array
    embedding_dim: int
    model: str
    enrollmentPictureThumbnail: str  # Base64 JPEG 128x128
    image_count: int
    processing_time_ms: int
    mock: bool = Field(default=False, description="True if response is from mock (no real DeepFace/embedding model)")


# ============================================
# Request/Response Models - VLM
# ============================================

class VLMImageInput(BaseModel):
    """Single image/frame input for VLM analysis"""
    frame: Optional[int] = None  # Frame index (for video)
    data: str  # Base64 encoded image


class VLMRequest(BaseModel):
    """Request for POST /iot/emotions"""
    event_id: Optional[str] = Field(default=None, description="Event identifier")
    employee_id: Optional[str] = Field(default=None, description="Employee identifier")
    images: List[VLMImageInput] = Field(..., description="Image frames to analyze")
    prompt: Optional[str] = Field(default=None, description="Custom analysis prompt")
    options: Optional[dict] = Field(default=None, description="Processing options")


class VLMResponse(BaseModel):
    """Response from POST /iot/emotions"""
    event_id: Optional[str] = None
    employee_id: Optional[str] = None
    emotions: List[dict]
    dominant_emotion: str
    analysis_summary: Optional[str] = None
    processing_time_ms: int
    mock: bool = Field(default=False, description="True if response is from mock backend (no real model)")


# ============================================
# Shared Models
# ============================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models: dict
    gpu_available: bool


# ============================================
# Lifespan Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize/cleanup resources"""
    global embedding_processor, emotion_processor

    logger.info("Starting GPU Server - initializing processors")

    try:
        # Initialize embedding processor
        embedding_config = config.get("models", {}).get("embedding", {})
        embedding_processor = EmbeddingProcessor(embedding_config)
        await embedding_processor.initialize()
        logger.info(f"Embedding processor initialized: {embedding_config.get('model_name', 'ArcFace')}")

        # Initialize emotion processor
        emotion_config = config.get("models", {}).get("emotion", {})
        emotion_processor = EmotionProcessor(emotion_config)
        await emotion_processor.initialize()
        logger.info(f"Emotion processor initialized: {emotion_config.get('backend', 'mock')}")

        logger.info(f"API routes loaded: {routes}")
        logger.info("GPU Server startup completed")
        yield

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        yield

    finally:
        logger.info("Shutting down GPU Server")
        if embedding_processor:
            await embedding_processor.cleanup()
        if emotion_processor:
            await emotion_processor.cleanup()


# ============================================
# FastAPI Application
# ============================================

logger.info("Creating FastAPI application")

# Gateway integration - set root_path from environment
# Incoming payloads are directed at /iot/(endpoint); gateway may add a prefix via ROOT_PATH.
ROOT_PATH = os.environ.get("ROOT_PATH", "")  # e.g., "" or "/galaxy/gpu"

app = FastAPI(
    title="GPU Processing Server",
    description="Facial Embedding & Emotion Analysis Service",
    version="1.0.0",
    lifespan=lifespan,
    root_path=ROOT_PATH,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Endpoints (under /iot/)
# ============================================

@app.get(routes["health"])
async def health():
    """Simple health check - IoT API base path"""
    return {"status": "ok"}


@app.get("/", response_model=HealthResponse)
@app.get(routes["models"], response_model=HealthResponse)
async def health_detail():
    """Detailed health check with model and GPU status"""
    logger.debug("Health check called")
    return HealthResponse(
        status="ok",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        models={
            "embedding": {
                "status": "ready" if embedding_processor and embedding_processor.is_ready else "not_ready",
                "model": config.get("models", {}).get("embedding", {}).get("model_name", "unknown")
            },
            "emotion": {
                "status": "ready" if emotion_processor and emotion_processor.is_ready else "not_ready",
                "backend": config.get("models", {}).get("emotion", {}).get("backend", "unknown")
            }
        },
        gpu_available=check_gpu_available()
    )


# ============================================
# Facial Recognition (vectorizer) - /iot/vectorizer
# ============================================

@app.post(routes["vectorizer"], response_model=VectorizerResponse)
async def facial_recognition_generate(request: VectorizerRequest):
    """
    Generate facial embedding from enrollment images.

    Called by: Enrollment Modal (via Bridge Server). IoT base path: /iot/vectorizer.
    Input: 1-5 pose images (front, left, right, up, down)
    Output: 512-dim ArcFace embedding + 128x128 thumbnail
    """
    request_id = f"vec-{str(uuid.uuid4())[:8]}"
    start_time = time.time()

    logger.info(
        f"[{request_id}] Vectorizer request - "
        f"employee_id: {request.employee_id}, images: {len(request.images)}"
    )

    if not embedding_processor or not embedding_processor.is_ready:
        raise HTTPException(status_code=503, detail="Embedding processor not ready")

    if len(request.images) < 1:
        raise HTTPException(status_code=400, detail="At least one image is required")

    try:
        # Decode images
        image_data = []
        for img in request.images:
            try:
                if img.data.startswith("data:"):
                    base64_data = img.data.split(",", 1)[1]
                else:
                    base64_data = img.data
                image_bytes = base64.b64decode(base64_data)
                image_data.append({"pose": img.pose, "bytes": image_bytes})
            except Exception as e:
                logger.error(f"[{request_id}] Failed to decode image: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        # Process
        result = await embedding_processor.process(
            employee_id=request.employee_id,
            images=image_data,
            options=request.options
        )

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"[{request_id}] Completed in {processing_time}ms")

        return VectorizerResponse(
            employee_id=request.employee_id,
            enrollmentProcessedFile=result["embedding_base64"],
            embedding_dim=result["embedding_dim"],
            model=result["model"],
            enrollmentPictureThumbnail=result["thumbnail_base64"],
            image_count=len(request.images),
            processing_time_ms=processing_time,
            mock="(mock)" in result.get("model", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Emotions (VLM) - /iot/emotions (vLLM runs internally in Docker)
# ============================================

@app.post(routes["emotions"], response_model=VLMResponse)
async def vlm_analyze(request: VLMRequest):
    """
    Analyze emotions in images using Vision Language Model (vLLM internal).

    Called by: Edge Devices (Jetson). IoT base path: /iot/emotions.
    Input: Image frames from video clip
    Output: Per-frame emotions, dominant emotion, narrative summary

    Planned: Pipeline will evolve to accept .mp4 (audio + full 15s video buffer);
    payload or delivery method may change when that pipeline is implemented.
    """
    request_id = f"vlm-{str(uuid.uuid4())[:8]}"
    start_time = time.time()

    logger.info(
        f"[{request_id}] VLM request - "
        f"event_id: {request.event_id}, images: {len(request.images)}"
    )

    if not emotion_processor or not emotion_processor.is_ready:
        raise HTTPException(status_code=503, detail="Emotion processor not ready")

    if len(request.images) < 1:
        raise HTTPException(status_code=400, detail="At least one image is required")

    try:
        # Decode images
        image_data = []
        for i, img in enumerate(request.images):
            try:
                if img.data.startswith("data:"):
                    base64_data = img.data.split(",", 1)[1]
                else:
                    base64_data = img.data
                image_bytes = base64.b64decode(base64_data)
                image_data.append({"frame": img.frame if img.frame is not None else i, "bytes": image_bytes})
            except Exception as e:
                logger.error(f"[{request_id}] Failed to decode image {i}: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid image data at index {i}: {e}")

        # Process
        result = await emotion_processor.process(
            employee_id=request.employee_id or request.event_id or "unknown",
            images=image_data,
            options=request.options
        )

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"[{request_id}] Completed in {processing_time}ms")

        return VLMResponse(
            event_id=request.event_id,
            employee_id=request.employee_id,
            emotions=result["emotions"],
            dominant_emotion=result["dominant_emotion"],
            analysis_summary=result.get("summary"),
            processing_time_ms=processing_time,
            mock=(emotion_processor.backend == "mock"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Transcription (Placeholder - TBD) - /iot/transcription
# ============================================

@app.post(routes["transcription"])
async def transcription_process(request: dict):
    """
    Audio transcription endpoint.

    Called by: TBD (additional user's code)
    Input: Audio data (format TBD)
    Output: Transcription results (format TBD)

    NOTE: Implementation pending - returns placeholder response
    """
    logger.warning("Transcription endpoint called but not yet implemented")
    return {
        "status": "not_implemented",
        "message": "Transcription functionality is not yet implemented",
        "note": "This endpoint is reserved for future audio transcription features"
    }


# ============================================
# Utility Functions
# ============================================

def check_gpu_available() -> bool:
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        pass

    return False


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 5000)

    logger.info(f"Starting GPU Server on {host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=server_config.get("workers", 1),
        reload=os.environ.get("DEV_MODE", "false").lower() == "true"
    )
