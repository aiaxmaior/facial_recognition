# GPU Server – API Endpoints & Payloads

## Health

### GET `/`
**Response:** `HealthResponse`
```json
{
  "status": "string",
  "timestamp": "string",
  "models": {},
  "gpu_available": true
}
```

### GET `/health`
**Response:** `{"status": "ok"}`

---

## Facial embedding (vectorizer)

**Endpoints:**  
`POST /v1/facial_recognition` · `POST /vectorizer/generate` · `POST /v1/vectorizer`

**Request:** `VectorizerRequest`
```json
{
  "employee_id": "string",
  "images": [
    { "pose": "front|left|right|up|down", "data": "<base64 image>" }
  ],
  "options": {}
}
```
- `employee_id`: required  
- `images`: 1–5 items, each `{ pose?, data }`, `data` = base64 image  
- `options`: optional

**Response:** `VectorizerResponse`
```json
{
  "employee_id": "string",
  "enrollmentProcessedFile": "<base64 Float32 embedding>",
  "embedding_dim": 512,
  "model": "ArcFace",
  "enrollmentPictureThumbnail": "<base64 JPEG 128x128>",
  "image_count": 5,
  "processing_time_ms": 0
}
```

---

## VLM emotion analysis

**Endpoints:**  
`POST /v1/vlm` · `POST /vlm/analyze`

**Request:** `VLMRequest`
```json
{
  "event_id": "string",
  "employee_id": "string",
  "images": [
    { "frame": 0, "data": "<base64 image>" }
  ],
  "prompt": "string",
  "options": {}
}
```
- `images`: required; each `{ frame?, data }`  
- `event_id`, `employee_id`, `prompt`, `options`: optional

**Response:** `VLMResponse`
```json
{
  "event_id": "string",
  "employee_id": "string",
  "emotions": [{}],
  "dominant_emotion": "string",
  "analysis_summary": "string",
  "processing_time_ms": 0
}
```

**Planned evolution (not yet in this package):**  
The VLM pipeline will be extended to accept an **.mp4 file** (or equivalent) containing **audio + full 15s video buffer**, for a combined vision + audio pipeline. Payload and/or delivery method (e.g. multipart, base64 video, or separate audio/video fields) may change when that pipeline is implemented.

---

## Transcription (placeholder)

**Endpoint:** `POST /v1/transcription`

**Request:** `{}` (body accepted as `dict`, format TBD)

**Response:**
```json
{
  "status": "not_implemented",
  "message": "Transcription functionality is not yet implemented",
  "note": "This endpoint is reserved for future audio transcription features"
}
```

---

## Logs

### GET `/logs`
**Query:** `limit` (optional, default 100)

**Response:**
```json
{
  "logs": [],
  "total": 0
}
```

### GET `/logs/view`
**Response:** HTML (log viewer UI)
