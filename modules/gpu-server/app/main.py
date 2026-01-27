"""
GPU Server - Facial Embedding & Emotion Analysis Service

Single Docker container serving two functions:
1. facial_embedding - Generate face embeddings using DeepFace/ArcFace
2. emotion_analysis - Analyze emotions using VLM

All requests go through POST /api/process with a "function" parameter.
"""

import os
import time
import uuid
import base64
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, List, Literal

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from processors.embedding import EmbeddingProcessor
from processors.emotion import EmotionProcessor
from utils.logging import LogHandler, get_log_viewer_html


# Load configuration
def load_config():
    config_path = os.environ.get("CONFIG_PATH", "/app/config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    # Default config
    return {
        "server": {"host": "0.0.0.0", "port": 5000, "root_path": "/api"},
        "models": {
            "embedding": {"model_name": "ArcFace", "embedding_dim": 512},
            "emotion": {"backend": "mock"}
        },
        "logging": {"level": "INFO", "max_log_entries": 1000}
    }


config = load_config()

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
# Request/Response Models
# ============================================

class ImageInput(BaseModel):
    """Single image input"""
    pose: Optional[str] = None  # front, left, right, up, down
    data: str  # Base64 encoded image


class ProcessRequest(BaseModel):
    """Main processing request"""
    function: Literal["facial_embedding", "emotion_analysis"] = Field(
        ..., description="Processing function to execute"
    )
    employee_id: str = Field(..., description="Employee/Person identifier")
    images: List[ImageInput] = Field(..., description="List of images to process")
    # Optional parameters
    options: Optional[dict] = Field(default=None, description="Function-specific options")


class EmbeddingResponse(BaseModel):
    """Response for facial_embedding function"""
    employee_id: str
    enrollmentProcessedFile: str  # Base64 Float32Array
    embedding_dim: int
    model: str
    enrollmentPictureThumbnail: str  # Base64 JPEG 128x128
    image_count: int
    processing_time_ms: int


class EmotionResponse(BaseModel):
    """Response for emotion_analysis function"""
    employee_id: str
    emotions: List[dict]  # [{frame, emotion, confidence, ...}]
    dominant_emotion: str
    analysis_summary: Optional[str] = None
    processing_time_ms: int


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
app = FastAPI(
    title="GPU Processing Server",
    description="Facial Embedding & Emotion Analysis Service",
    version="1.0.0",
    lifespan=lifespan,
    root_path=config.get("server", {}).get("root_path", "/api")
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
# Endpoints
# ============================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
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


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "ok"}


@app.post("/process")
async def process(request: ProcessRequest):
    """
    Main processing endpoint - routes to appropriate function.
    
    Functions:
    - facial_embedding: Generate face embeddings from images
    - emotion_analysis: Analyze emotions in images/video frames
    """
    request_id = f"req-{str(uuid.uuid4())[:8]}"
    start_time = time.time()
    
    logger.info(
        f"Processing request {request_id} - function: {request.function}, "
        f"employee_id: {request.employee_id}, images: {len(request.images)}"
    )
    
    try:
        if request.function == "facial_embedding":
            result = await process_embedding(request, request_id)
        elif request.function == "emotion_analysis":
            result = await process_emotion(request, request_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown function: {request.function}")
        
        processing_time = int((time.time() - start_time) * 1000)
        result["processing_time_ms"] = processing_time
        
        logger.info(f"Request {request_id} completed in {processing_time}ms")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_embedding(request: ProcessRequest, request_id: str) -> dict:
    """Process facial embedding request"""
    if not embedding_processor or not embedding_processor.is_ready:
        raise HTTPException(status_code=503, detail="Embedding processor not ready")
    
    if len(request.images) < 1:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    logger.debug(f"Request {request_id}: Processing {len(request.images)} images for embedding")
    
    # Decode images
    image_data = []
    for img in request.images:
        try:
            # Handle data URL format or raw base64
            if img.data.startswith("data:"):
                base64_data = img.data.split(",", 1)[1]
            else:
                base64_data = img.data
            image_bytes = base64.b64decode(base64_data)
            image_data.append({"pose": img.pose, "bytes": image_bytes})
        except Exception as e:
            logger.error(f"Request {request_id}: Failed to decode image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    # Process with embedding processor
    result = await embedding_processor.process(
        employee_id=request.employee_id,
        images=image_data,
        options=request.options
    )
    
    return {
        "employee_id": request.employee_id,
        "enrollmentProcessedFile": result["embedding_base64"],
        "embedding_dim": result["embedding_dim"],
        "model": result["model"],
        "enrollmentPictureThumbnail": result["thumbnail_base64"],
        "image_count": len(request.images),
    }


async def process_emotion(request: ProcessRequest, request_id: str) -> dict:
    """Process emotion analysis request"""
    if not emotion_processor or not emotion_processor.is_ready:
        raise HTTPException(status_code=503, detail="Emotion processor not ready")
    
    if len(request.images) < 1:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    logger.debug(f"Request {request_id}: Processing {len(request.images)} images for emotion analysis")
    
    # Decode images
    image_data = []
    for i, img in enumerate(request.images):
        try:
            if img.data.startswith("data:"):
                base64_data = img.data.split(",", 1)[1]
            else:
                base64_data = img.data
            image_bytes = base64.b64decode(base64_data)
            image_data.append({"frame": i, "bytes": image_bytes})
        except Exception as e:
            logger.error(f"Request {request_id}: Failed to decode image {i}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data at index {i}: {e}")
    
    # Process with emotion processor
    result = await emotion_processor.process(
        employee_id=request.employee_id,
        images=image_data,
        options=request.options
    )
    
    return {
        "employee_id": request.employee_id,
        "emotions": result["emotions"],
        "dominant_emotion": result["dominant_emotion"],
        "analysis_summary": result.get("summary"),
    }


# ============================================
# Logging Endpoints (following reference pattern)
# ============================================

@app.get("/logs")
async def get_logs(limit: int = 100):
    """API endpoint to retrieve logs"""
    logger.debug(f"Logs endpoint called with limit={limit}")
    logs = list(log_storage)
    return {"logs": logs[-limit:], "total": len(logs)}


@app.get("/logs/view", response_class=HTMLResponse)
async def view_logs():
    """Web UI to view logs in real-time"""
    return HTMLResponse(content=get_log_viewer_html())


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    websocket_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total connections: {len(websocket_connections)}")
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


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
        "main:app",
        host=host,
        port=port,
        workers=server_config.get("workers", 1),
        reload=os.environ.get("DEV_MODE", "false").lower() == "true"
    )
