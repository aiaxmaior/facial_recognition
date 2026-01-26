# Facial Recognition System
Arjun Joshi
1.26.25

A distributed facial recognition and emotion monitoring platform with edge device deployment, centralized GPU processing, and IoT broker integration.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        BRIDGE SERVER                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Facial Enrollment Modal                        │  │
│  │         (React Frontend + Node.js API Server)              │  │
│  │                                                             │  │
│  │  Receives: POST from Employee Management Web Page          │  │
│  │  Sends:    → IoT Broker (publish enrollment)               │  │
│  │            → GPU Server (generate embedding)               │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                │                              │
                │                              │
                ▼                              ▼
┌───────────────────────┐        ┌───────────────────────┐
│      IoT Broker       │        │      GPU Server       │
│    (Data Service)     │        │    (AWS/Vectorizer)   │
│                       │        │                       │
│  • Device Registry    │        │  • ArcFace Embeddings │
│  • Event Storage      │        │  • VLM Story Gen      │
│  • WebSocket Broadcast│        │  • Thumbnail Gen      │
└───────────┬───────────┘        └───────────────────────┘
            │
            │ WebSocket (enrollment_update, events)
            ▼
┌───────────────────────────────────────────────────────────┐
│                    Edge Devices                            │
│              (Jetson Orin Nano + RTSP Camera)             │
│                                                            │
│  • Face Recognition    • Emotion Monitoring               │
│  • Video Clip Capture  • Event Transmission               │
└───────────────────────────────────────────────────────────┘
```

## Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Bridge Server** | Hosts enrollment modal, connects to IoT & GPU | `facial-enrollment-modal/` |
| **GPU Server** | Facial embedding vectorization (ArcFace), VLM | `gpu-server/` |
| **Jetson Edge** | Edge device runtime for Orin Nano | `jetson_edge/` |
| **IoT Integration** | Client library for broker communication | `iot_integration/` |
| **Video Analysis** | Emotion detection pipeline | `video_analysis/` |
| **Core Library** | DeepFace-based recognition system | `facial_recognition.py` |

## Quick Start

### GPU Server (Local Testing)

```bash
cd gpu-server
docker compose build
docker compose up
```

API available at `http://localhost:5000/api`

### Jetson Edge Device

```bash
# Initial setup (run once)
sudo ./jetson_edge/scripts/setup_jetson_headless.sh

# Install service
sudo ./jetson_edge/scripts/install_service.sh

# Start service
sudo systemctl start qraie-facial
```

### Standalone Recognition

```bash
pip install -r requirements.txt

# Enroll faces
python facial_recognition.py enroll -n "John Doe" -i photo1.jpg photo2.jpg

# Match against database
python facial_recognition.py match -t unknown.jpg

# Launch web interface
python facial_recognition.py --interface
```

## API Documentation

### Event Message Format (Socket.IO)

All events use a header + data structure:

```json
[
  {
    "header": {
      "to": "gateway",
      "from": "cam-001",
      "source_type": "device",
      "auth_token": "...",
      "command_id": "event.log",
      "timestamp": "2026-01-23T10:30:00.000Z"
    }
  },
  {
    "data": {
      "event_id": "EV1-1706005800000-cam001",
      "event_type": "face_recognition",
      "person_name": "John Doe",
      "person_id": "EMP-001",
      "metadata": { "confidence": 0.95 },
      "debug": []
    }
  }
]
```

### Event Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `face_recognition` | Face identified | `person_name`, `person_id`, `metadata.confidence` |
| `emotion_monitoring` | Emotion analysis | `story`, `video_clip`, `metadata.duration` |

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `debug` | Array | Graylog entries (REQUIRED, can be empty `[]`) |
| `person_name` | String | Display name (face_recognition only, cannot be empty) |
| `person_id` | String | Employee ID (face_recognition only) |

### Data Formats

| Data | Format | Description |
|------|--------|-------------|
| Embedding | Base64 Float32Array | 512-dimension vector |
| Thumbnail | Base64 JPEG | 128x128 pixels |
| Video Clip | Base64 MP4 | 15 seconds (10s pre + 5s post event) |
| Story | String | VLM narrative (~100 tokens) |

## API Endpoints

### IoT Broker (v1)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/events` | POST | Submit events (Socket.IO format) |
| `/api/v1/events/video` | POST | Submit emotion event with video clip |
| `/api/data/devices` | GET/POST | Device registration |
| `/api/data/devices/{id}/heartbeat` | POST | Device heartbeat |
| `/api/data/enrollment/publish` | POST | Publish enrollment to devices |
| `/api/data/enrollments/sync` | GET | Pull enrollment updates |

### GPU Server

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/vectorizer/generate` | POST | Generate facial embedding |
| `/api/embed` | POST | Generate embedding (legacy) |

See [OpenAPI specs](docs/swagger/) for full API documentation:
- `edge-device-api.yaml` - Edge device communication
- `enrollment-api.yaml` - Enrollment system

## Enrollment Flow

```
1. User clicks "Enroll Face" in WFM Dashboard
2. Modal opens, captures 5 pose images (front, left, right, up, down)
3. Modal sends images to GPU Server → receives embedding + thumbnail
4. Modal publishes to IoT Broker → broker broadcasts to edge devices
5. Modal stores enrollment in Bridge database
```

**Note:** The Modal receives confirmation that the IoT Broker *sent* the enrollment, not that edge devices *received* it. This is a "fire and forget" model. Edge devices sync missing enrollments via pull-based sync.

## Project Structure

```
facial_recognition/
├── gpu-server/                  # GPU vectorization service (Docker)
│   ├── app/
│   │   ├── main.py              # FastAPI server
│   │   └── processors/          # Embedding & emotion processors
│   └── docker-compose.yaml
├── jetson_edge/                 # Edge device deployment
│   ├── main.py                  # Main entry point
│   ├── video_buffer.py          # 15-second ring buffer
│   ├── scripts/                 # Setup scripts
│   └── systemd/                 # Service files
├── iot_integration/             # IoT client library
│   ├── iot_client.py            # Broker communication
│   ├── event_validator.py       # Multi-frame confirmation
│   ├── image_utils.py           # Image/video encoding
│   └── schemas/                 # Pydantic models
│       ├── event_schemas.py     # Event payloads
│       └── sync_schemas.py      # Enrollment sync
├── facial-enrollment-modal/     # Web enrollment UI (in Bridge)
│   └── packages/
│       ├── api-server/          # Node.js backend
│       └── web-client/          # React frontend
├── video_analysis/              # Emotion analysis pipeline
│   ├── scripts/
│   │   ├── emotion_detector.py
│   │   └── person_detector.py
│   └── notebooks/
├── docs/                        # API documentation
│   ├── swagger/                 # OpenAPI specs
│   │   ├── edge-device-api.yaml
│   │   └── enrollment-api.yaml
│   ├── EDGE_DEVICE_PROTOCOL.md
│   └── IOT_BROKER_FRAMEWORK.md
├── facial_recognition.py        # Core recognition system
├── face_admin.py                # Database admin tool
└── requirements.txt
```

## Hardware

- **GPU Server**: CUDA-capable GPU (RTX series recommended)
- **Edge Devices**: NVIDIA Jetson Orin Nano
- **Cameras**: RTSP IP cameras (Reolink, etc.)

## Documentation

- [Edge Device Protocol](docs/EDGE_DEVICE_PROTOCOL.md) - API specs and deployment
- [IoT Broker Framework](docs/IOT_BROKER_FRAMEWORK.md) - Backend architecture
- [Enrollment API Spec](docs/swagger/enrollment-api.yaml) - OpenAPI specification
- [Edge Device API Spec](docs/swagger/edge-device-api.yaml) - OpenAPI specification
- [Jetson Edge README](jetson_edge/README.md) - Edge device setup
- [GPU Server README](gpu-server/README.md) - Vectorization service
- [Reolink Setup](reolink_jetson_setup.md) - Camera configuration

## License

Proprietary
