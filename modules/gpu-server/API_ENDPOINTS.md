# GPU Server API Endpoints

Public API uses the base path **`/iot/(endpoint)`** only (no `/v1`). vLLM runs internally in Docker.

## IoT Endpoints (3 Functions)

### 1. Facial Recognition (Embedding Generation)
**Endpoint:** `POST /iot/vectorizer`

**Purpose:** Generate facial embeddings from enrollment images
**Called by:** Enrollment Modal (via Bridge Server)
**Input:** 1-5 pose images (front, left, right, up, down)
**Output:** 512-dim ArcFace embedding + 128x128 thumbnail

**Request:**
```json
{
  "employee_id": "EMP123456",
  "images": [
    {"pose": "front", "data": "base64..."},
    {"pose": "left", "data": "base64..."},
    {"pose": "right", "data": "base64..."}
  ],
  "options": {}
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
  "image_count": 3,
  "processing_time_ms": 1250
}
```

---

### 2. Emotions (VLM – runs internally)
**Endpoint:** `POST /iot/emotions`

**Purpose:** Analyze emotions in images using Vision Language Model
**Called by:** Edge Devices (Jetson)
**Input:** Image frames from video clip
**Output:** Per-frame emotions, dominant emotion, narrative summary

**Request:**
```json
{
  "event_id": "EV2-1706005800000-jetson001",
  "employee_id": "EMP123456",
  "images": [
    {"frame": 0, "data": "base64..."},
    {"frame": 5, "data": "base64..."},
    {"frame": 10, "data": "base64..."}
  ],
  "prompt": "Analyze the emotion in these images",
  "options": {}
}
```

**Response:**
```json
{
  "event_id": "EV2-1706005800000-jetson001",
  "employee_id": "EMP123456",
  "emotions": [
    {"frame": 0, "emotion": "neutral", "confidence": 0.85},
    {"frame": 5, "emotion": "happiness", "confidence": 0.92},
    {"frame": 10, "emotion": "surprise", "confidence": 0.78}
  ],
  "dominant_emotion": "happiness",
  "analysis_summary": "Person appears happy and engaged",
  "processing_time_ms": 2340
}
```

---

### 3. Transcription (Placeholder - TBD)
**Endpoint:** `POST /iot/transcription`

**Purpose:** Audio transcription (future implementation)
**Called by:** TBD (additional user's code)
**Input:** Audio data (format TBD)
**Output:** Transcription results (format TBD)

**Current Response:**
```json
{
  "status": "not_implemented",
  "message": "Transcription functionality is not yet implemented",
  "note": "This endpoint is reserved for future audio transcription features"
}
```

**TODO:** Implement when requirements are defined

---

## System Endpoints

### Health Check (Simple)
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

---

### Health Check (Detailed)
**Endpoint:** `GET /`

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-02-10T12:34:56Z",
  "models": {
    "embedding": {
      "status": "ready",
      "model": "ArcFace"
    },
    "emotion": {
      "status": "ready",
      "backend": "vllm"
    }
  },
  "gpu_available": true
}
```

---

### Logs (JSON API)
**Endpoint:** `GET /logs?limit=100`

**Response:**
```json
{
  "logs": [
    "2026-02-10 12:34:56 | INFO | Server started",
    "2026-02-10 12:35:00 | INFO | Request received"
  ],
  "total": 245
}
```

---

### Logs (Web UI)
**Endpoint:** `GET /logs/view`

Returns HTML interface for viewing logs in real-time

---

### Logs (WebSocket Stream)
**Endpoint:** `WS /ws/logs`

WebSocket connection for real-time log streaming

---

## Gateway Integration

When deployed behind gateway with `ROOT_PATH=/galaxy/gpu`:

| Function | Endpoint | Full URL |
|----------|----------|----------|
| Facial Recognition | `/iot/vectorizer` | `https://gateway.com/galaxy/gpu/iot/vectorizer` |
| Emotions (VLM) | `/iot/emotions` | `https://gateway.com/galaxy/gpu/iot/emotions` |
| Transcription | `/iot/transcription` | `https://gateway.com/galaxy/gpu/iot/transcription` |
| Health (simple) | `/iot/health` | `https://gateway.com/galaxy/gpu/iot/health` |
| Models (detail) | `/iot/models` | `https://gateway.com/galaxy/gpu/iot/models` |
| Logs UI | `/iot/logs/view` | `https://gateway.com/galaxy/gpu/iot/logs/view` |

---

## OpenAPI/Swagger Docs

**Endpoint:** `GET /docs`
**With gateway:** `https://gateway.com/galaxy/gpu/docs`

Interactive API documentation with:
- Request/response schemas
- Try-it-out functionality
- Model definitions

---

## CORS Configuration

Currently allows all origins (`*`) for development.

**For production:** Update `app/main.py` line 194:
```python
allow_origins=[
    "https://yourdomain.com",
    "https://enrollment.yourdomain.com",
    os.environ.get("ALLOWED_ORIGIN", "*")
]
```

---

## Authentication

**Current:** None (open endpoints)

**Recommended for AWS:**
- VPC isolation (security groups)
- API keys via headers
- Gateway-level authentication

---

## Rate Limiting

**Current:** None

**Recommended:** Configure at gateway level or add via FastAPI middleware

---

## Summary

**Active Endpoints:** 3 functions under `/iot/` + health/logs
**API base:** `/iot/(endpoint)` only (no `/v1`)
**Gateway Ready:** ROOT_PATH configurable
**Future:** Transcription endpoint placeholder ready
