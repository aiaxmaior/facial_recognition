# Edge Device <-> IoT Broker Protocol

> Current as of 2026-02-17. Documents the exact protocol the edge device
> implements for communication with the IoT broker.

---

## Overview

| Channel | Transport | Direction | Purpose |
|---------|-----------|-----------|---------|
| **Socket.IO** | `'message'` event | Both | All real-time communication |
| **HTTP REST** | POST | Edge -> Broker | Events, heartbeats, video clips |

All Socket.IO messages use a single event name **`'message'`** with a
two-element array payload: `[header, data]`.

Routing is determined by `header.command_id`.

---

## 1. Socket.IO Connection

**Server:** `https://acetaxi-bridge.qryde.net`
**Path:** `/iot-broker/api/socket.io` (Socket.IO path behind reverse proxy)
**Namespace:** `/` (default)

The `python-socketio` client handles:
- Engine.IO handshake and upgrade
- Ping/pong keepalive
- Automatic reconnection with exponential backoff (1s to 60s)

---

## 2. Message Format

Every message is emitted as a single Socket.IO `'message'` event with a
two-element array payload. Each element is wrapped in a key:

```python
sio.emit('message', [{"header": {...}}, {"data": {...}}])
```

### Header (first element)

```json
{
  "header": {
    "to": "gateway",
    "from": "jetson-001",
    "source_type": "device",
    "auth_token": null,
    "command_id": "device.register",
    "timestamp": "2026-02-17T12:00:00.000Z"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `to` | string | Destination (`"gateway"` for broker-bound) |
| `from` | string | Sender identifier (device_id or `"gateway"`) |
| `source_type` | string | `"device"` or `"gateway"` |
| `auth_token` | string or null | JWT, API key, or `"gateway"` |
| `command_id` | string | **Routing key** -- determines message type |
| `timestamp` | string | ISO 8601 UTC |

### Routing by command_id

| command_id | Direction | Purpose |
|---|---|---|
| `device.register` | Edge -> Broker | Register device on connect |
| `device.heartbeat` | Edge -> Broker | Heartbeat over Socket.IO (broker then updates data-service) |
| `response.success` | Broker -> Edge | Confirm registration |
| `enrollment.publish` | Broker -> Edge | Push a face embedding |
| `event.log` | Edge -> Broker | Send recognition/emotion event |

---

## 3. Device Registration (Edge -> Broker)

Sent immediately after Socket.IO connects.

```json
[
  {
    "header": {
      "to": "gateway",
      "from": "jetson-001",
      "source_type": "device",
      "auth_token": null,
      "command_id": "device.register",
      "timestamp": "2026-02-17T12:00:00.000Z"
    }
  },
  {
    "data": {
      "request_id": "550e8400-e29b-41d4-a716-446655440000",
      "device_category": "camera",
      "capability": "face_recognition"
    }
  }
]
```

### Broker confirms with `response.success`:

```json
[
  {
    "header": {
      "to": "jetson-001",
      "from": "gateway",
      "source_type": "gateway",
      "auth_token": "gateway",
      "command_id": "response.success",
      "timestamp": "2026-02-17T12:00:00.050Z"
    }
  },
  {
    "data": { }
  }
]
```

---

## 4. Enrollment Push (Broker -> Edge)

**This is the key message.** Sent when a user is enrolled and the broker
has a new face embedding to distribute to devices.

```json
[
  {
    "header": {
      "to": "jetson-001",
      "from": "gateway",
      "source_type": "gateway",
      "auth_token": "gateway",
      "command_id": "enrollment.publish",
      "timestamp": "2026-02-17T12:05:00.000Z"
    }
  },
  {
    "data": {
      "employee_id": "E693",
      "person_id": "E693",
      "person_name": "John Doe",
      "embedded_file": "<base64-encoded float32 array>",
      "enrollment_data": {}
    }
  }
]
```

### What the device stores

Only two fields matter:

| Field | Description |
|-------|-------------|
| `employee_id` | Primary key in local SQLite |
| `embedded_file` | Base64-encoded raw bytes of a float32 array (512 floats for ArcFace = 2048 bytes) |

All other fields (`person_id`, `person_name`, `enrollment_data`) are
accepted but **not persisted** -- for anonymity and simplicity.

### Decoding `embedded_file`

```python
import base64, numpy as np

raw_bytes = base64.b64decode(embedded_file)
embedding = np.frombuffer(raw_bytes, dtype=np.float32)  # shape: (512,)
```

### No acknowledgment

The device does **not** send an ack for enrollment messages.

### What happens on the device

1. Decode `embedded_file` to numpy array
2. Upsert `(employee_id, embedding)` into local SQLite
3. Reload all embeddings into the recognition pipeline (hot-reload)
4. Next frame processed uses the updated embedding set

---

## 5. Face Recognition Events (Edge -> Broker)

Sent via HTTP POST and also use the `[{"header": {...}}, {"data": {...}}]` format.

```
POST https://acetaxi-bridge.qryde.net/iot-broker/api/data/events
```

```json
[
  {
    "header": {
      "to": "gateway",
      "from": "jetson-001",
      "source_type": "device",
      "auth_token": null,
      "command_id": "event.log",
      "timestamp": "2026-02-17T12:10:00.000Z"
    }
  },
  {
    "data": {
      "event_id": "EV1-1739793000000-tson-001",
      "event_type": "face_recognition",
      "person_name": "John Doe",
      "person_id": "EMP-123",
      "metadata": {
        "confidence": 0.95,
      "person_detected": true,
      "distance": 0.42,
      "frames_tracked": 3,
      "inference_ms": 45,
      "face_bbox": [100, 150, 200, 250]
    },
      "debug": [],
      "image": "<base64 JPEG, max 50KB>"
    }
  }
]
```

Headers:
- `Content-Type: application/json`
- `X-Device-ID: jetson-001`

---

## 6. Heartbeat (Edge -> Broker)

```
POST https://acetaxi-bridge.qryde.net/iot-broker/api/data/devices/jetson-001/heartbeat
```

Sent every 30 seconds.

```json
{
  "timestamp": "2026-02-17T12:01:00.000Z",
  "status": "operational",
  "metrics": {
    "cpu_percent": 45.2,
    "memory_percent": 62.1,
    "temperature_c": 48.5,
    "uptime_seconds": 3600
  },
  "cv_stats": {
    "fps_current": 4.8,
    "frames_processed": 17280,
    "detections_count": 42,
    "recognitions_count": 15
  },
  "queue_depth": 0
}
```

---

## 7. Current Device Configuration

```json
{
  "device_id": "jetson-001",
  "broker_url": "https://acetaxi-bridge.qryde.net/iot-broker/api",
  "api_key": null,
  "sync": {
    "enrollment_db_path": "data/enrollments.db",
    "socketio_url": "https://acetaxi-bridge.qryde.net",
    "socketio_path": "/iot-broker/api/socket.io"
  },
  "heartbeat": {
    "interval_seconds": 30
  }
}
```
