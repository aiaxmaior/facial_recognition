# Edge Device - Facial Recognition Service

Real-time facial recognition and event processing runtime for **Jetson Orin Nano** edge devices. Captures video from RTSP cameras, performs face detection/recognition against an enrolled database, validates events, and transmits them to the IoT broker.

## Table of Contents

- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Scripts](#scripts)
- [IoT Integration](#iot-integration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Jetson Orin Nano                                  │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────────┐    │
│  │ RTSP Camera  │───▶│  Detection   │───▶│  Recognition Pipeline      │    │
│  │ (Reolink)    │    │  (OpenCV/    │    │  (DeepFace + ArcFace)      │    │
│  │              │    │   YOLOv8)    │    │                            │    │
│  └──────────────┘    └──────────────┘    └─────────────┬──────────────┘    │
│                                                        │                    │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────▼──────────────┐    │
│  │ Video Ring   │◀───│  Event       │◀───│  Enrollment Database       │    │
│  │ Buffer       │    │  Validator   │    │  (SQLite)                  │    │
│  └──────────────┘    └──────────────┘    └────────────────────────────┘    │
│         │                   │                                               │
│         │                   ▼                                               │
│         │            ┌──────────────┐                                       │
│         └───────────▶│  IoT Client  │                                       │
│                      │  (HTTP/WS)   │                                       │
│                      └──────┬───────┘                                       │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ IoT Broker  │  │ GPU Server  │  │ Enrollment  │
     │ (Events)    │  │ (VLM)       │  │ Sync        │
     └─────────────┘  └─────────────┘  └─────────────┘
```

---

## Directory Structure

```
edge-device/
├── config/                     # Configuration files
│   ├── config.json             # Active device configuration
│   └── config_example.json     # Template configuration
├── data/                       # Runtime data
│   └── enrollments.db          # Local enrollment database (SQLite)
├── iot_integration/            # IoT broker client library
│   ├── iot_client.py           # HTTP/WebSocket client
│   ├── event_validator.py      # Event validation logic
│   ├── sync_manager.py         # Enrollment synchronization
│   ├── db_manager.py           # SQLite database manager
│   ├── image_utils.py          # Image encoding utilities
│   └── schemas/                # Pydantic data models
│       ├── event_schemas.py    # Event payloads
│       ├── db_schemas.py       # Database models
│       └── sync_schemas.py     # Sync protocols
├── scripts/                    # Utility scripts
│   ├── discover_camera.py      # Auto-find cameras on network
│   ├── install_service.sh      # Install as systemd service
│   ├── setup_jetson_headless.sh# Initial Jetson setup
│   └── migrate_enrollments.py  # Database migration tool
├── src/                        # Main application source
│   ├── main.py                 # Main entry point
│   ├── device_ctl.py           # Device control CLI
│   ├── video_buffer.py         # Ring buffer for video clips
│   └── test_event.py           # Event testing utility
├── systemd/                    # Service files
│   └── qraie-facial.service    # Systemd unit file
├── video_analysis/             # Emotion detection (optional)
│   ├── emotion_detector.py
│   ├── person_detector.py
│   └── video_preprocessor.py
├── test_images/                # Test assets
├── activate.sh                 # Activate Python venv
├── requirements.txt            # Python dependencies
└── edge-device-api.yaml        # OpenAPI specification
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Face Recognition** | Real-time identification using DeepFace + ArcFace model |
| **Event Validation** | Multi-frame confirmation to reduce false positives |
| **Video Capture** | 15-second ring buffer (10s pre + 5s post event) |
| **IoT Integration** | Heartbeat, event transmission, enrollment sync |
| **Camera Discovery** | Auto-find Reolink cameras on the network |
| **Visual Debugging** | Optional display mode with bounding boxes |
| **Systemd Service** | Production deployment with auto-restart |

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **Compute** | NVIDIA Jetson Orin Nano (8GB recommended) |
| **Camera** | RTSP IP Camera (Reolink or similar) |
| **Network** | Ethernet connection to camera and IoT broker |
| **Storage** | 32GB+ SD card or NVMe |

---

## Quick Start

```bash
# 1. Navigate to edge-device directory
cd /home/qdrive/facial_recognition/modules/edge-device

# 2. Activate Python virtual environment
source activate.sh

# 3. Discover and configure camera
python scripts/discover_camera.py --config config/config.json

# 4. Check device status
python src/device_ctl.py status --config config/config.json

# 5. Run the service
python src/main.py --config config/config.json

# Or with visual debugging (requires display)
python src/main.py --config config/config.json --display
```

---

## Installation

### Development Setup

```bash
# Clone/navigate to project
cd /home/qdrive/facial_recognition/modules/edge-device

# Activate virtual environment
source activate.sh

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Production Deployment (systemd)

```bash
# Install as system service
sudo ./scripts/install_service.sh

# Enable and start
sudo systemctl enable qraie-facial
sudo systemctl start qraie-facial

# Check status
sudo systemctl status qraie-facial

# View logs
sudo journalctl -u qraie-facial -f
```

### Initial Jetson Setup

For fresh Jetson devices:

```bash
sudo ./scripts/setup_jetson_headless.sh
```

---

## Configuration

### Configuration File

Edit `config/config.json`:

```json
{
    "device_id": "jetson-001",
    "broker_url": "https://acetaxi-bridge.qryde.net/iot-broker/api",
    "api_key": null,
    "camera": {
        "rtsp_url": "rtsp://admin:password@192.168.13.119/Preview_01_main",
        "fps": 25
    },
    "recognition": {
        "model": "ArcFace",
        "detector_backend": "opencv",
        "distance_threshold": 0.35,
        "process_fps": 1,
        "process_width": 640
    },
    "validation": {
        "confirmation_frames": 5,
        "consistency_threshold": 0.8,
        "cooldown_seconds": 30
    },
    "video_buffer": {
        "enabled": false,
        "duration_seconds": 15,
        "buffer_path": "/tmp/video_buffer",
        "pre_event_seconds": 10,
        "post_event_seconds": 5
    },
    "sync": {
        "enrollment_db_path": "/path/to/enrollments.db"
    },
    "heartbeat": {
        "interval_seconds": 30
    }
}
```

### Configuration Options

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| **device_id** | - | Unique device identifier | `jetson-001` |
| **broker_url** | - | IoT broker API endpoint | - |
| **camera.rtsp_url** | - | RTSP stream URL | - |
| **camera.fps** | - | Camera frame rate | `25` |
| **recognition.model** | - | Face recognition model | `ArcFace` |
| **recognition.detector_backend** | - | Face detector (`opencv`, `retinaface`, `yolov8`) | `opencv` |
| **recognition.distance_threshold** | - | Match threshold (lower = stricter) | `0.35` |
| **recognition.process_fps** | - | Frames to process per second | `1` |
| **recognition.process_width** | - | Resize width for processing | `640` |
| **validation.confirmation_frames** | - | Frames needed to confirm identity | `5` |
| **validation.consistency_threshold** | - | Required consistency ratio | `0.8` |
| **validation.cooldown_seconds** | - | Cooldown between same-person events | `30` |
| **video_buffer.enabled** | - | Enable video clip capture | `false` |
| **heartbeat.interval_seconds** | - | Heartbeat frequency | `30` |

---

## Usage

### Device Control CLI

The `device_ctl.py` script provides management commands:

```bash
# Show full device status (broker, camera, system metrics)
python src/device_ctl.py status --config config/config.json

# Register device with IoT broker
python src/device_ctl.py register --config config/config.json

# Send single heartbeat
python src/device_ctl.py heartbeat --config config/config.json

# Continuous heartbeat (every 30s)
python src/device_ctl.py heartbeat --loop --config config/config.json

# Auto-configure camera (interactive)
python src/device_ctl.py camera --config config/config.json

# Full startup (register + heartbeat + run)
python src/device_ctl.py run --config config/config.json
```

### Main Service

```bash
# Basic run
python src/main.py --config config/config.json

# With visual display (bounding boxes, FPS overlay)
python src/main.py --config config/config.json --display

# Press 'q' to quit when display is active
```

### Test Events

```bash
# Send a test face recognition event
python src/test_event.py --config config/config.json
```

---

## Scripts

### Camera Discovery (`scripts/discover_camera.py`)

Automatically finds Reolink cameras on the network and updates configuration:

```bash
# Auto-discover and update config
python scripts/discover_camera.py --config config/config.json

# Dry run (show what would be done)
python scripts/discover_camera.py --dry-run

# Scan specific subnet
python scripts/discover_camera.py --subnet 192.168.13

# Custom credentials
python scripts/discover_camera.py --username admin --password mypassword

# Specify preferred stream
python scripts/discover_camera.py --stream Preview_01_main
```

### Service Installation (`scripts/install_service.sh`)

Installs the application as a systemd service:

```bash
sudo ./scripts/install_service.sh
```

This will:
- Create `qraie` system user
- Copy files to `/opt/qraie/`
- Install systemd service
- Create config at `/opt/qraie/config/device_config.json`

### Enrollment Migration (`scripts/migrate_enrollments.py`)

Migrate enrollment data between database versions:

```bash
python scripts/migrate_enrollments.py --source old.db --target new.db
```

---

## IoT Integration

### Event Flow

1. **Face Detected** → Detection pipeline finds face in frame
2. **Face Recognized** → ArcFace matches against enrollment database
3. **Event Validated** → Multiple consistent frames confirm identity
4. **Event Transmitted** → IoT client sends to broker via HTTP/WebSocket

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/data/devices` | POST | Register device |
| `/data/devices/{id}/heartbeat` | POST | Send heartbeat |
| `/data/events` | POST | Submit recognition events |
| `/data/events/video` | POST | Submit events with video clip |
| `/data/enrollments/sync` | GET | Pull enrollment updates |

### Event Payload Format

Events are sent as JSON arrays (Socket.IO format):

```json
[
  {
    "header": {
      "to": "gateway",
      "from": "jetson-001",
      "command_id": "event.log",
      "timestamp": "2026-01-30T10:30:00.000Z"
    }
  },
  {
    "data": {
      "event_id": "EV1-1706005800000-jetson001",
      "event_type": "face_recognition",
      "person_name": "John Doe",
      "person_id": "john_doe",
      "metadata": {
        "confidence": 0.95,
        "distance": 0.28
      },
      "debug": []
    }
  }
]
```

### Heartbeat Format

```json
{
  "status": "operational",
  "metrics": {
    "cpu_percent": 45.3,
    "memory_percent": 67.2,
    "temperature_c": 50.4
  }
}
```

---

## API Reference

### Device Registration

```bash
curl -X POST "https://acetaxi-bridge.qryde.net/iot-broker/api/data/devices" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "jetson-001",
    "display_name": "Jetson Orin - Front Desk",
    "capability": "face_recognition",
    "status": "provisioning",
    "device_category": "camera",
    "location_label": "Office Entrance"
  }'
```

### Heartbeat

```bash
curl -X POST "https://acetaxi-bridge.qryde.net/iot-broker/api/data/devices/jetson-001/heartbeat" \
  -H "Content-Type: application/json" \
  -H "X-Device-ID: jetson-001" \
  -d '{
    "status": "operational",
    "metrics": {
      "cpu_percent": 45.0,
      "memory_percent": 60.0,
      "temperature_c": 48.5
    }
  }'
```

### Check Device Status

```bash
curl "https://acetaxi-bridge.qryde.net/iot-broker/api/data/devices" \
  -H "X-Device-ID: jetson-001"
```

---

## Troubleshooting

### Camera Not Found

```bash
# 1. Check camera is on same network
ip addr show | grep "inet "

# 2. Scan for cameras
python scripts/discover_camera.py --dry-run

# 3. Test RTSP directly
ffprobe -v quiet "rtsp://admin:password@192.168.13.119/Preview_01_main"
```

### Broker Connection Failed

```bash
# 1. Check network connectivity
curl -v https://acetaxi-bridge.qryde.net/iot-broker/api/health

# 2. Verify DNS resolution
host acetaxi-bridge.qryde.net

# 3. Check device status
python src/device_ctl.py status --config config/config.json
```

### Low FPS / Slow Recognition

1. Reduce `process_width` in config (e.g., 480 instead of 640)
2. Use `opencv` detector instead of `retinaface`
3. Decrease `process_fps` (e.g., 0.5 for 1 frame every 2 seconds)
4. Disable video buffer if not needed

### Recognition Accuracy Issues

1. Increase `distance_threshold` (e.g., 0.4) for looser matching
2. Increase `confirmation_frames` for more validation
3. Ensure good lighting and camera angle
4. Re-enroll users with better quality images

### Service Won't Start

```bash
# Check logs
sudo journalctl -u qraie-facial -n 50 --no-pager

# Verify config
python src/device_ctl.py status --config /opt/qraie/config/device_config.json

# Test manually
sudo -u qraie /opt/qraie/facial_recognition/venv/bin/python \
  /opt/qraie/facial_recognition/main.py \
  --config /opt/qraie/config/device_config.json
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device index | `0` |
| `CONFIG_PATH` | Path to config file | `/opt/qraie/config/device_config.json` |
| `PYTHONUNBUFFERED` | Unbuffered output | `1` |

---

## License

Proprietary - QRaie Technologies

---

## Support

For issues and questions:
- Check [Troubleshooting](#troubleshooting) section
- Review logs: `sudo journalctl -u qraie-facial -f`
- Contact the development team
