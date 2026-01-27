# IoT Integration Module

This module provides integration between the facial recognition edge devices (Jetson Orin Nano) and the central WFM dashboard via an IoT broker.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Central Dashboard (WFM)                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   MongoDB    │    │  Backend API │    │  Dashboard   │       │
│  │ (Enrollments)│    │   (Events)   │    │     UI       │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       IoT Broker                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Sync      │    │    Events    │    │  Heartbeat   │       │
│  │   Endpoint   │    │   Endpoint   │    │   Endpoint   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS (PULL)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Edge Device (Jetson Orin)                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   SQLite     │    │    Event     │    │     IoT      │       │
│  │ (Local DB)   │◄───│  Validator   │───►│    Client    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ▲                   ▲                                    │
│         │                   │                                    │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │    Sync      │    │   Face       │                           │
│  │   Manager    │    │ Recognition  │                           │
│  └──────────────┘    └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Database Manager (`db_manager.py`)

Handles local SQLite storage with the new user_id-based schema:

```python
from iot_integration import DatabaseManager

db = DatabaseManager("/path/to/faces.db", dev_mode=False)
db.initialize()

# Add/update enrollment
db.upsert_enrollment("EMP-123", embedding_array, "ArcFace")

# Get all embeddings for matching
embeddings = db.get_all_enrollments()  # {user_id: np.array}
```

### 2. Event Validator (`event_validator.py`)

Validates recognition events by tracking across frames:

```python
from iot_integration import EventValidator

validator = EventValidator(
    device_id="cam-001",
    confirmation_frames=5,       # Need 5 consecutive frames
    consistency_threshold=0.8,   # 80% must be same user
    cooldown_seconds=30          # 30s between same-user events
)

# For each recognition result:
event = validator.process_recognition(
    track_id="face_0",
    user_id="EMP-123",
    distance=0.28,
    frame_id=100
)

if event:
    # Event validated! Send to broker
    iot_client.send_event(event)
```

### 3. IoT Client (`iot_client.py`)

Handles communication with the IoT broker:

```python
from iot_integration import IoTClient, IoTClientConfig

config = IoTClientConfig(
    device_id="cam-001",
    broker_url="https://iot-broker.example.com",
    api_key="your-api-key"
)

client = IoTClient(config)
client.start()

# Send events (async, queued)
client.send_event(event, image=frame, face_bbox=[x, y, w, h])

# Or synchronous
client.send_event_sync(event)

# Request enrollment sync
response = client.request_sync(since_version=100)

client.stop()
```

### 4. Sync Manager (`sync_manager.py`)

Manages enrollment database synchronization:

```python
from iot_integration import SyncManager, SyncManagerFactory

# Easy setup with factory
sync_mgr, db, client = SyncManagerFactory.create(
    db_path="/path/to/faces.db",
    device_id="cam-001",
    broker_url="https://iot-broker.example.com"
)

# Start periodic sync (every 15 minutes)
sync_mgr.start_periodic_sync(interval_minutes=15)

# Or manual sync
sync_mgr.sync_now()

# Check status
status = sync_mgr.get_sync_status()
```

### 5. Image Utils (`image_utils.py`)

Utilities for image compression and encoding:

```python
from iot_integration import compress_image_for_event

# Compress image for event payload (target: 50KB JPEG)
b64_image = compress_image_for_event(
    frame,
    target_size_kb=50,
    face_bbox=[x, y, w, h]
)
```

## Event Payloads

### Facial ID Event

```json
{
    "event_id": "EV1-1736942400000-a1b2c3",
    "device_id": "cam-001",
    "user_id": "EMP-12345",
    "event_type": "facial_id",
    "timestamp": "2026-01-15T10:00:00.000Z",
    "confidence": 0.92,
    "image": "<base64-jpeg>",
    "metadata": {
        "distance": 0.28,
        "frames_tracked": 5,
        "inference_ms": 45
    }
}
```

### Emotion Event

```json
{
    "event_id": "EV2.2-1736942400000-d4e5f6",
    "device_id": "cam-001",
    "user_id": "EMP-12345",
    "event_type": "emotion",
    "emotion_code": "EV2.2",
    "timestamp": "2026-01-15T10:00:05.000Z",
    "confidence": 0.88,
    "image": "<base64-jpeg>",
    "metadata": {
        "intensity": 0.75,
        "duration_ms": 2500
    }
}
```

## SQLite Schema

### enrolled_users

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| user_id | TEXT | Central WFM employee ID (unique) |
| display_name | TEXT | Dev mode only |
| model | TEXT | Embedding model (e.g., "ArcFace") |
| detector | TEXT | Face detector used |
| embedding | BLOB | Float32 embedding bytes |
| embedding_dim | INTEGER | Embedding dimension |
| sync_version | INTEGER | Version when synced |
| synced_at | TEXT | ISO timestamp |
| created_at | TEXT | ISO timestamp |

### device_config

| Column | Type | Description |
|--------|------|-------------|
| key | TEXT | Configuration key (PK) |
| value | TEXT | Configuration value |
| updated_at | TEXT | ISO timestamp |

## Configuration

See `config_example.json` for a complete configuration template.

Key settings:

- `confirmation_frames`: Number of consecutive frames needed to validate an event (default: 5)
- `consistency_threshold`: Ratio of frames that must have the same user_id (default: 0.8)
- `cooldown_seconds`: Time before same user can trigger new event (default: 30)
- `dev_mode`: If true, display_name is stored locally for debugging

## Development Mode

When `dev_mode=True`:
- Display names are stored in local SQLite (for debugging)
- Additional logging is enabled
- Local enrollment via the enrollment module is allowed

When `dev_mode=False` (production):
- Only user_id is stored locally
- All enrollments come from central database via sync
- PII is not stored on edge devices
