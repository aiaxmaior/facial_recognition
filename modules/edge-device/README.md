# Edge Device

Facial recognition and emotion monitoring runtime for **Jetson Orin Nano** edge devices.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Jetson Orin Nano                             │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ RTSP Camera  │───▶│  Detection   │───▶│  Recognition     │   │
│  │ (Reolink)    │    │  Pipeline    │    │  Pipeline        │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────▼─────────┐   │
│  │ Video Ring   │◀───│  Event       │◀───│  IoT Client      │   │
│  │ Buffer       │    │  Validator   │    │                  │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
                          ┌─────────────────────────┼──────────────┐
                          ▼                         ▼              │
              ┌─────────────────┐       ┌─────────────────┐        │
              │   GPU Server    │       │   IoT Broker    │        │
              │   (VLM)         │       │   (Events)      │        │
              └─────────────────┘       └─────────────────┘        │
```

## Components

| Directory | Description |
|-----------|-------------|
| `src/` | Main application source |
| `iot_integration/` | IoT broker client library |
| `video_analysis/` | Emotion detection scripts |
| `scripts/` | Setup and installation scripts |
| `systemd/` | Service files |
| `config/` | Configuration examples |

## Features

- **Face Recognition**: Real-time identification against enrolled database
- **Emotion Monitoring**: Emotional state analysis via VLM
- **Video Capture**: 15-second ring buffer (10s pre + 5s post event)
- **IoT Integration**: Event transmission, heartbeat, enrollment sync

## Hardware Requirements

- NVIDIA Jetson Orin Nano (8GB recommended)
- RTSP IP Camera (Reolink or similar)
- Network connectivity to IoT Broker

## Installation

```bash
# Run setup script
sudo ./scripts/setup_jetson_headless.sh

# Install as service
sudo ./scripts/install_service.sh
```

## Configuration

Edit `config/config.json`:

```json
{
    "device_id": "jetson-001",
    "broker_url": "https://acetaxi-bridge.qryde.net/iot-broker/api",
    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
    "heartbeat": {
        "interval_seconds": 30
    }
}
```

## API Communication

### To IoT Broker

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/data/devices` | POST | Register device |
| `/data/devices/{id}/heartbeat` | POST | Send heartbeat (every 30s) |
| `/data/events` | POST | Submit recognition events |
| `/data/events/video` | POST | Submit emotion events with video |
| `/data/enrollments/sync` | GET | Pull enrollment updates |

### To GPU Server

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vlm/analyze` | POST | Emotion analysis via VLM |

## Event Format (Socket.IO)

```json
[
  {
    "header": {
      "to": "gateway",
      "from": "jetson-001",
      "command_id": "event.log",
      "timestamp": "2026-01-23T10:30:00.000Z"
    }
  },
  {
    "data": {
      "event_id": "EV1-1706005800000-jetson001",
      "event_type": "face_recognition",
      "person_name": "John Doe",
      "person_id": "EMP-001",
      "metadata": { "confidence": 0.95 },
      "debug": []
    }
  }
]
```

## Running

```bash
# Manual run
python src/main.py --config config/config.json

# As service
sudo systemctl start qraie-facial
sudo systemctl status qraie-facial

# View logs
sudo journalctl -u qraie-facial -f
```

## Device Registration

```bash
# Register device with IoT Broker
curl -X POST "https://acetaxi-bridge.qryde.net/iot-broker/api/data/devices" \
  -H "Content-Type: application/json" \
  -d '{"device_id":"jetson-001","display_name":"Jetson Orin","capability":"face_recognition","status":"provisioning"}'

# Send heartbeat
curl -X POST "https://acetaxi-bridge.qryde.net/iot-broker/api/data/devices/jetson-001/heartbeat" \
  -H "Content-Type: application/json" \
  -d '{"status":"operational"}'
```
