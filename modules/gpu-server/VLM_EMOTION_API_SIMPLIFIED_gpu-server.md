# GPU Server – API Summary

Brief overview of what the service exposes and what each endpoint expects/returns.

---

## Health

| Endpoint | What it does |
|----------|----------------|
| **GET** `/` | Detailed health (status, models, GPU). |
| **GET** `/health` | Simple check; returns `{"status":"ok"}`. |

No request body. Response is JSON.

---

## Facial enrollment (vectorizer)

**Endpoints:**  
`POST /iot/vectorizer`

**Purpose:** Turn 1–5 face photos into a single embedding + thumbnail for enrollment.

**How the vectorizer behaves:** The vectorizer does not send the embedding or thumbnail to any other API. It only returns them in the HTTP response to the client that called the endpoint. The caller (e.g. an enrollment API server) is responsible for storing the result, forwarding it to an IoT broker or bridge, or returning it to a UI. The GPU server is strictly request/response: no outbound callbacks or push to third-party URLs.

**Request (JSON):**
- `employee_id` – who is being enrolled (required)
- `images` – list of 1–5 images; each has optional `pose` (e.g. `"front"`, `"left"`) and `data` (base64 image)

**Response (JSON):**
- `employee_id`, `embedding_dim` (512), `model` ("ArcFace")
- `enrollmentProcessedFile` – base64 embedding (for storage/matching)
- `enrollmentPictureThumbnail` – base64 128×128 JPEG
- `image_count`, `processing_time_ms`

---

## Emotion analysis (VLM)

**Endpoints:**  
`POST /iot/emotions`

**Purpose:** Analyze emotions in a set of image frames (e.g. from a short clip).

**Request (JSON):**
- `images` – list of images; each has optional `frame` index and `data` (base64 image). At least one image required.
- Optional: `event_id`, `employee_id`, `prompt`, `options`

**Response (JSON):**
- `emotions` (per-frame), `dominant_emotion`, `analysis_summary`
- `event_id`, `employee_id` (echoed if sent), `processing_time_ms`

---

## Transcription

**Endpoint:** `POST /iot/transcription`

**Status:** Placeholder only. Returns “not implemented.” Reserved for future audio transcription.

---

## Logs

| Endpoint | What it does |
|----------|----------------|
| **GET** `/logs` | Returns recent log entries (JSON). Optional query: `?limit=100`. |
| **GET** `/logs/view` | Web page to view logs in a browser. |

No request body for these.
