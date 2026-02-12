# GPU Server

Centralized GPU processing server for facial embedding generation and VLM-based emotion analysis. Runs on **AWS GPU instance** via Docker.

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      AWS GPU Server                            │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Server                        │  │
│  │                                                          │  │
│  │  ┌─────────────────────┐    ┌─────────────────────────┐ │  │
│  │  │   Vectorizer        │    │   VLM (vLLM)            │ │  │
│  │  │   (ArcFace)         │    │   (Emotion Analysis)    │ │  │
│  │  │                     │    │                         │ │  │
│  │  │  POST /iot/         │    │  POST /iot/              │ │  │
│  │  │  vectorizer         │    │  emotions                │ │  │
│  │  └─────────────────────┘    └─────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                    ▲                         ▲
                    │                         │
        ┌───────────┴───────────┐   ┌────────┴────────┐
        │  Enrollment Modal     │   │  Edge Devices   │
        │  (Bridge Server)      │   │  (Jetson)       │
        └───────────────────────┘   └─────────────────┘
```

## API Endpoints (base path `/iot/`)

| Endpoint | Method | Caller | Description |
|----------|--------|--------|-------------|
| `/iot/vectorizer` | POST | Enrollment Modal | Generate facial embedding |
| `/iot/emotions` | POST | Edge Device | Emotion analysis (VLM internal) |
| `/iot/health` | GET | Any | Health check |

## Components

| File | Description |
|------|-------------|
| `app/main.py` | FastAPI application |
| `app/processors/embedding.py` | ArcFace embedding generator |
| `app/processors/emotion.py` | VLM emotion analyzer |
| `Dockerfile` | Docker build configuration |
| `docker-compose.yaml` | Docker Compose setup |
| `config.yaml` | Server configuration |

## API Details

### Vectorizer (Enrollment Modal → GPU Server)

**Request:**
```json
{
  "employee_id": "EMP123456",
  "images": [
    {"pose": "front", "data": "base64..."},
    {"pose": "left", "data": "base64..."},
    {"pose": "right", "data": "base64..."},
    {"pose": "up", "data": "base64..."},
    {"pose": "down", "data": "base64..."}
  ],
  "function": "facial_embedding"
}
```

**Response:**
```json
{
  "employee_id": "EMP123456",
  "enrollmentProcessedFile": "base64_float32_array...",
  "embedding_dim": 512,
  "model": "ArcFace",
  "enrollmentPictureThumbnail": "base64_jpeg_128x128...",
  "image_count": 5
}
```

### VLM Analysis (Edge Device → GPU Server)

**Request:**
```json
{
  "event_id": "EV2-1706005800000-jetson001",
  "video_clip": "base64_mp4...",
  "metadata": {
    "duration": 15.0,
    "fps": 25
  }
}
```

**Response:**
```json
{
  "event_id": "EV2-1706005800000-jetson001",
  "story": "A person entered the room looking happy and engaged...",
  "inference_ms": 1250
}
```

## Installation

### Docker (Recommended)

```bash
cd modules/gpu-server
docker compose build
docker compose up -d
```

### Manual

```bash
pip install -r requirements.txt
python app/main.py
```

## Configuration

Edit `config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 5000

models:
  embedding:
    name: ArcFace
    backend: tensorrt  # or onnx
  vlm:
    name: Qwen/Qwen3-VL-8B-Thinking
    backend: vllm

gpu:
  device: cuda:0
  memory_fraction: 0.8
```

## Docker Compose

```yaml
services:
  gpu-server:
    build: .
    ports:
      - "5000:5000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./config.yaml:/app/config.yaml
```

## Health Check

```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "gpu": "available",
  "models": {
    "embedding": "loaded",
    "vlm": "loaded"
  }
}
```

## Hardware Requirements

- NVIDIA GPU with 8GB+ VRAM (RTX 3080/4080 or better)
- CUDA 11.8+
- 32GB+ RAM
- Docker with NVIDIA Container Toolkit
