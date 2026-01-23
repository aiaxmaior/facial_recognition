# Jetson Edge Device

Facial Recognition Service for Jetson Orin Nano.

## Overview

This directory contains the edge device software for running facial recognition on NVIDIA Jetson Orin Nano. The service:

- Captures video from RTSP camera
- Detects and recognizes faces using DeepFace/ArcFace
- Validates events across multiple frames
- Transmits events to IoT broker
- Maintains 15-second video buffer for event clips

## Directory Structure

```
jetson_edge/
├── main.py              # Main entry point
├── video_buffer.py      # Video ring buffer for clip capture
├── config/              # Configuration templates
├── scripts/
│   ├── setup_jetson_headless.sh   # Jetson setup script
│   └── install_service.sh         # Service installer
└── systemd/
    └── qraie-facial.service       # Systemd service file
```

## Quick Start

### 1. Setup Jetson (run once)

```bash
# Basic setup (disables GUI, creates directories)
sudo ./scripts/setup_jetson_headless.sh

# Or with full GUI removal (saves 3-5GB)
sudo ./scripts/setup_jetson_headless.sh --full-removal
```

### 2. Install Service

```bash
sudo ./scripts/install_service.sh
```

### 3. Configure

```bash
sudo nano /opt/qraie/config/device_config.json
```

Set your:
- `device_id`: Unique identifier
- `camera.rtsp_url`: Camera RTSP URL
- `broker_url`: IoT broker endpoint

### 4. Start

```bash
sudo systemctl enable qraie-facial
sudo systemctl start qraie-facial
```

## Configuration

See `/opt/qraie/config/device_config.json`:

| Section | Parameter | Description |
|---------|-----------|-------------|
| `camera` | `rtsp_url` | Camera RTSP stream URL |
| `camera` | `fps` | Target frame rate (default: 25) |
| `recognition` | `distance_threshold` | Max cosine distance for match (default: 0.35) |
| `validation` | `confirmation_frames` | Frames required to confirm (default: 5) |
| `validation` | `cooldown_seconds` | Seconds between same-user events (default: 30) |
| `video_buffer` | `enabled` | Enable video clip capture |
| `video_buffer` | `duration_seconds` | Total buffer duration (default: 15) |

## Service Management

```bash
# Status
sudo systemctl status qraie-facial

# Logs
sudo journalctl -u qraie-facial -f

# Restart
sudo systemctl restart qraie-facial
```

## Development

### Run locally (without systemd)

```bash
# Activate venv
source /opt/qraie/facial_recognition/venv/bin/activate

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run
python main.py --config /opt/qraie/config/device_config.json --debug
```

### Dependencies

- Python 3.10+
- OpenCV with GStreamer support
- DeepFace
- NVIDIA CUDA / TensorRT (for GPU acceleration)

## Architecture

```
Camera (RTSP)
    │
    ▼
┌─────────────────────────────────────┐
│           Frame Capture             │
│         (GStreamer/FFmpeg)          │
└───────────────┬─────────────────────┘
                │
                ├──────────────────────┐
                │                      │
                ▼                      ▼
┌───────────────────────┐   ┌─────────────────────┐
│   Face Detection      │   │   Video Buffer      │
│   (YOLOv8/RetinaFace) │   │   (15s ring)        │
└───────────┬───────────┘   └──────────┬──────────┘
            │                          │
            ▼                          │
┌───────────────────────┐              │
│   Face Recognition    │              │
│   (ArcFace 512-dim)   │              │
└───────────┬───────────┘              │
            │                          │
            ▼                          │
┌───────────────────────┐              │
│   Event Validator     │──────────────┤
│   (5-frame confirm)   │              │
└───────────┬───────────┘              │
            │                          │
            ▼                          ▼
┌───────────────────────┐   ┌─────────────────────┐
│     IoT Client        │   │    Clip Encoder     │
│   (REST/WebSocket)    │   │    (H.265 NVENC)    │
└───────────┬───────────┘   └──────────┬──────────┘
            │                          │
            ▼                          ▼
      IoT Broker                Archive Server
```

## Troubleshooting

| Issue | Check |
|-------|-------|
| Camera not connecting | `ffprobe rtsp://...` |
| No faces detected | Lighting, camera angle, min_face_size |
| Recognition failing | Distance threshold, enrollment sync |
| Service crashes | `journalctl -u qraie-facial -n 100` |
| GPU not used | `nvidia-smi`, CUDA_VISIBLE_DEVICES |
