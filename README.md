# QRAIE Facial Recognition System

A distributed facial recognition and emotion monitoring platform with three independent modules deployed across different infrastructure.

**Author:** Arjun Joshi  
**Version:** 2.0.0  
**Date:** 2026-01-27

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                           ┌─────────────────────┐                           │
│                           │    IoT Broker       │                           │
│                           │  (Central Hub)      │                           │
│                           └──────────┬──────────┘                           │
│                                      │                                       │
│            ┌─────────────────────────┼─────────────────────────┐            │
│            │                         │                         │            │
│            ▼                         ▼                         ▼            │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐     │
│  │  ENROLLMENT     │      │   EDGE DEVICE   │      │   GPU SERVER    │     │
│  │  MODAL          │      │                 │      │                 │     │
│  │                 │      │  Jetson Orin    │      │  AWS/Docker     │     │
│  │  Bridge Server  │      │  Nano           │      │                 │     │
│  │                 │      │                 │      │  • Vectorizer   │     │
│  │  • Enrollment   │      │  • Face Recog   │      │  • VLM          │     │
│  │  • Webcam UI    │      │  • Emotion Mon  │      │                 │     │
│  │  • Publish      │      │  • Video Clip   │      │                 │     │
│  └────────┬────────┘      └────────┬────────┘      └────────┬────────┘     │
│           │                        │                        │               │
│           │   ┌────────────────────┴────────────────────┐   │               │
│           │   │                                         │   │               │
│           └───┴─────────── API Calls ───────────────────┴───┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Location | Deployment | Description |
|--------|----------|------------|-------------|
| **Enrollment Modal** | `modules/enrollment-modal/` | Bridge Server | Web-based face enrollment interface |
| **Edge Device** | `modules/edge-device/` | Jetson Orin Nano | Real-time recognition & emotion monitoring |
| **GPU Server** | `modules/gpu-server/` | AWS (Docker) | Centralized ML processing |

## Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ENROLLMENT FLOW                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Enrollment Modal ──[images]──▶ GPU Server ──[embedding]──▶ IoT Broker      │
│                                                                  │           │
│                                                                  ▼           │
│                                                           Edge Devices       │
│                                                          (via WebSocket)     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ RECOGNITION FLOW                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Edge Device ──[event]──▶ IoT Broker                                        │
│       │                                                                      │
│       └──[video clip]──▶ GPU Server (VLM) ──[story]──▶ IoT Broker           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### IoT Broker
Base URL: `https://{tenant}-bridge.qryde.net/iot-broker/api`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/data/devices` | POST | Register device |
| `/data/devices/{id}/heartbeat` | POST | Device heartbeat (30s) |
| `/data/events` | POST | Submit events |
| `/data/events/video` | POST | Submit emotion event + video |
| `/data/enrollment/publish` | POST | Publish enrollment |
| `/data/enrollments/sync` | GET | Pull enrollment updates |

### GPU Server

| Endpoint | Method | Caller | Description |
|----------|--------|--------|-------------|
| `/api/vectorizer/generate` | POST | Enrollment Modal | Generate embedding |
| `/api/vlm/analyze` | POST | Edge Device | Emotion analysis |

## Event Format (Socket.IO)

```json
[
  { "header": { "to": "gateway", "from": "device-id", "command_id": "event.log", "timestamp": "..." } },
  { "data": { "event_id": "...", "event_type": "face_recognition", "person_name": "...", "person_id": "...", "debug": [] } }
]
```

## Quick Start

### Enrollment Modal (Bridge Server)
```bash
cd modules/enrollment-modal
npm install
npm run dev
```

### Edge Device (Jetson)
```bash
cd modules/edge-device
sudo ./scripts/setup_jetson_headless.sh
sudo ./scripts/install_service.sh
```

### GPU Server (Docker)
```bash
cd modules/gpu-server
docker compose build
docker compose up -d
```

## Documentation

| Document | Location |
|----------|----------|
| Edge Device API | `modules/docs/swagger/edge-device-api.yaml` |
| Enrollment API | `modules/docs/swagger/enrollment-api.yaml` |
| IoT Broker Framework | `modules/docs/IOT_BROKER_FRAMEWORK.md` |
| Edge Protocol | `modules/docs/EDGE_DEVICE_PROTOCOL.md` |

## Directory Structure

```
facial_recognition/
├── modules/
│   ├── enrollment-modal/     # Bridge server component
│   │   ├── packages/
│   │   │   ├── api-server/   # Node.js backend
│   │   │   └── react-component/  # React UI
│   │   ├── audio/            # Guidance audio
│   │   └── README.md
│   │
│   ├── edge-device/          # Jetson deployment
│   │   ├── src/              # Main application
│   │   ├── iot_integration/  # IoT client library
│   │   ├── video_analysis/   # Emotion detection
│   │   ├── scripts/          # Setup scripts
│   │   ├── systemd/          # Service files
│   │   ├── config/           # Configuration
│   │   └── README.md
│   │
│   ├── gpu-server/           # AWS GPU processing
│   │   ├── app/              # FastAPI application
│   │   ├── Dockerfile
│   │   ├── docker-compose.yaml
│   │   └── README.md
│   │
│   └── docs/                 # Shared documentation
│       └── swagger/          # OpenAPI specs
│
├── .gitignore
└── README.md
```

## License

Proprietary - QRAIE Systems
