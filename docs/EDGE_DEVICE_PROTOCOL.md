# Edge Device Communication Protocol

**Version:** 1.0  
**Last Updated:** 2026-01-22

This document defines the communication protocols for Edge Devices (Jetson Orin Nano) in the Facial Recognition System, covering event submission, enrollment synchronization, video archival, and log shipping.

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication & Device Identity](#authentication--device-identity)
3. [Edge to IoT Broker API](#edge-to-iot-broker-api)
4. [IoT Broker to Edge WebSocket Protocol](#iot-broker-to-edge-websocket-protocol)
5. [Edge to Archive Server API](#edge-to-archive-server-api)
6. [Recognition Validation Workflow](#recognition-validation-workflow)
7. [Video Buffer & Clip Capture](#video-buffer--clip-capture)
8. [Log Shipping Protocol](#log-shipping-protocol)
9. [Data Types Reference](#data-types-reference)
10. [Error Handling & Retry Strategy](#error-handling--retry-strategy)
11. [Jetson Deployment Guide](#jetson-deployment-guide)

---

## Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EDGE DEVICE COMMUNICATION                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────────┐                           ┌──────────────────┐          │
│   │                  │      REST (HTTPS)         │                  │          │
│   │   Edge Device    │ ─────────────────────────►│   IoT Broker     │          │
│   │   (Jetson Orin)  │   Events, Heartbeat       │   Data Service   │          │
│   │                  │                           │                  │          │
│   │  ┌────────────┐  │      WebSocket (WSS)      │                  │          │
│   │  │ Face       │  │ ◄─────────────────────────│                  │          │
│   │  │ Recognition│  │   Enrollments, Commands   │                  │          │
│   │  └────────────┘  │                           └──────────────────┘          │
│   │                  │                                                          │
│   │  ┌────────────┐  │      REST (HTTPS)         ┌──────────────────┐          │
│   │  │ Video      │  │ ─────────────────────────►│                  │          │
│   │  │ Buffer     │  │   Clips, Logs (Batch)     │  Archive Server  │          │
│   │  └────────────┘  │                           │                  │          │
│   │                  │                           └──────────────────┘          │
│   └──────────────────┘                                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Communication Summary

| Direction | Protocol | Purpose | Frequency |
|-----------|----------|---------|-----------|
| Edge → IoT Broker | REST/HTTPS | Event submission | Per validated event |
| Edge → IoT Broker | REST/HTTPS | Heartbeat | Every 30 seconds |
| IoT Broker → Edge | WebSocket | Enrollment updates | On enrollment change |
| IoT Broker → Edge | WebSocket | Commands | On demand |
| Edge → Archive | REST/HTTPS | Video clips | Per event (optional) |
| Edge → Archive | REST/HTTPS | Log batches | Every 30s or 100 entries |

### Service Endpoints

| Service | Base URL | Purpose |
|---------|----------|---------|
| IoT Broker - Data | `https://{tenant}-bridge{env}.qraie.ai/api/data` | Events, Devices |
| IoT Broker - WebSocket | `wss://{tenant}-bridge{env}.qraie.ai/ws/device` | Real-time sync |
| Archive Server | `https://archive.qraie.ai/api/archive` | Clips, Logs |

---

## Authentication & Device Identity

### Device Registration

Each edge device must be registered before operation. Registration provides:
- Unique `device_id`
- Device certificate for mTLS (optional)
- API token for REST calls

### Request Headers

All REST requests from edge devices must include:

```http
Host: {tenant}-bridge{env}.qraie.ai
Content-Type: application/json
X-Device-ID: {device_id}
Authorization: Bearer {device_token}
```

### WebSocket Authentication

WebSocket connections authenticate via query parameters or headers:

```
wss://{tenant}-bridgestg.qraie.ai/ws/device?device_id=cam-001&token={device_token}
```

Or via headers during upgrade:
```http
X-Device-ID: cam-001
Authorization: Bearer {device_token}
```

---

## Edge to IoT Broker API

### Submit Events (Batched)

Submit recognition and/or emotion events in batches for efficiency.

```http
POST /api/data/events
Host: {tenant}-bridgestg.qraie.ai
Content-Type: application/json
X-Device-ID: cam-001
```

**Request Body:**
```json
{
  "device_id": "cam-001",
  "batch_id": "batch-1737550200-001",
  "timestamp": "2026-01-22T10:30:00.000Z",
  "events": [
    {
      "event_id": "EV1-1737550200123-a1b2c3d4",
      "event_type": "face_recognition",
      "timestamp": "2026-01-22T10:30:00.123Z",
      "user_id": "EMP123456",
      "confidence": 0.94,
      "metadata": {
        "distance": 0.28,
        "frames_tracked": 5,
        "inference_ms": 42,
        "face_bbox": [100, 50, 200, 200]
      },
      "image": "base64_encoded_jpeg..."
    },
    {
      "event_id": "EV2.2-1737550200456-b2c3d4e5",
      "event_type": "emotion",
      "timestamp": "2026-01-22T10:30:00.456Z",
      "user_id": "EMP123456",
      "emotion_code": "EV2.2",
      "confidence": 0.87,
      "metadata": {
        "intensity": 0.87,
        "duration_ms": 750
      }
    }
  ]
}
```

**Response (Success - 200):**
```json
{
  "success": true,
  "batch_id": "batch-1737550200-001",
  "accepted": 2,
  "rejected": 0
}
```

**Response (Partial Success - 207):**
```json
{
  "success": true,
  "batch_id": "batch-1737550200-001",
  "accepted": 1,
  "rejected": 1,
  "errors": [
    {
      "event_id": "EV2.2-1737550200456-b2c3d4e5",
      "error": "Unknown user_id"
    }
  ]
}
```

---

### Device Heartbeat

Update device status and receive configuration updates.

```http
POST /api/data/devices/{device_id}/heartbeat
Host: {tenant}-bridgestg.qraie.ai
```

**Request Body:**
```json
{
  "timestamp": "2026-01-22T10:30:00Z",
  "status": "operational",
  "metrics": {
    "cpu_percent": 45.2,
    "memory_percent": 62.1,
    "gpu_memory_mb": 1024,
    "temperature_c": 52.0,
    "disk_percent": 30.5,
    "uptime_seconds": 86400
  },
  "cv_stats": {
    "fps_current": 28.5,
    "fps_average": 27.2,
    "frames_processed": 184700,
    "detections_count": 2340,
    "recognitions_count": 1890,
    "avg_inference_ms": 45.2
  },
  "buffer_status": {
    "video_buffer_seconds": 15,
    "video_buffer_size_mb": 45.2,
    "event_queue_depth": 3,
    "log_queue_depth": 127
  },
  "sync_status": {
    "enrollment_version": 1547,
    "enrolled_count": 2341,
    "last_sync_at": "2026-01-22T00:00:00Z"
  }
}
```

**Response (Success - 200):**
```json
{
  "success": true,
  "server_time": "2026-01-22T10:30:01Z",
  "config_version": "v2.3",
  "config_update_available": false,
  "enrollment_version": 1547
}
```

**Response (Config Update Available):**
```json
{
  "success": true,
  "server_time": "2026-01-22T10:30:01Z",
  "config_version": "v2.4",
  "config_update_available": true,
  "config_url": "/api/data/devices/cam-001/config"
}
```

---

### Request Enrollment Sync

Pull enrollment updates since a known version.

```http
GET /api/data/enrollments/sync?device_id=cam-001&since_version=1540&model=ArcFace
Host: {tenant}-bridgestg.qraie.ai
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `device_id` | string | Yes | Device identifier |
| `since_version` | integer | No | Last known sync version (default: 0) |
| `model` | string | No | Expected embedding model (default: ArcFace) |
| `force_full` | boolean | No | Force full database sync |

**Response (Success - 200):**
```json
{
  "sync_version": 1547,
  "total_enrolled": 2341,
  "additions": [
    {
      "user_id": "EMP123456",
      "embedding": "base64_encoded_float32_array...",
      "dim": 512,
      "model": "ArcFace",
      "display_name": "John Doe"
    },
    {
      "user_id": "EMP123457",
      "embedding": "base64_encoded_float32_array...",
      "dim": 512,
      "model": "ArcFace",
      "display_name": "Jane Smith"
    }
  ],
  "removals": ["EMP100001", "EMP100002"],
  "full_sync_required": false,
  "synced_at": "2026-01-22T10:30:00Z"
}
```

---

## IoT Broker to Edge WebSocket Protocol

### Connection Lifecycle

```
1. Edge connects to WebSocket endpoint
2. Edge sends 'register' message with device_id
3. Broker sends 'registered' acknowledgment
4. Broker pushes messages as needed
5. Edge sends 'ack' for messages requiring confirmation
6. Edge sends 'ping' periodically (every 30s)
7. Broker responds with 'pong'
```

### Message Envelope

All WebSocket messages follow this envelope structure:

```json
{
  "type": "message_type",
  "message_id": "msg-uuid-12345",
  "timestamp": "2026-01-22T10:30:00Z",
  "require_ack": true,
  "payload": { ... }
}
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `register` | Edge → Broker | Device registration on connect |
| `registered` | Broker → Edge | Registration acknowledgment |
| `enrollment_update` | Broker → Edge | New/updated enrollment |
| `enrollment_delete` | Broker → Edge | Remove enrollment |
| `command` | Broker → Edge | Device command |
| `config_update` | Broker → Edge | Configuration change |
| `ack` | Edge → Broker | Message acknowledgment |
| `ping` | Edge → Broker | Keep-alive ping |
| `pong` | Broker → Edge | Keep-alive response |

---

### Registration Message

Sent immediately after WebSocket connection established.

**Edge → Broker:**
```json
{
  "type": "register",
  "message_id": "reg-001",
  "timestamp": "2026-01-22T10:30:00Z",
  "payload": {
    "device_id": "cam-001",
    "device_type": "jetson_orin_nano",
    "capability": "face_recognition",
    "firmware_version": "2.1.0",
    "enrollment_version": 1540,
    "model": "ArcFace"
  }
}
```

**Broker → Edge:**
```json
{
  "type": "registered",
  "message_id": "reg-001",
  "timestamp": "2026-01-22T10:30:00.050Z",
  "payload": {
    "status": "success",
    "server_time": "2026-01-22T10:30:00.050Z",
    "enrollment_version": 1547,
    "sync_required": true
  }
}
```

---

### Enrollment Update Message

Pushed when a new enrollment is published or updated.

**Broker → Edge:**
```json
{
  "type": "enrollment_update",
  "message_id": "enroll-update-12345",
  "timestamp": "2026-01-22T10:30:00Z",
  "require_ack": true,
  "payload": {
    "action": "upsert",
    "employee_id": "EMP123456",
    "person_name": "John Doe",
    "embedded_file": "base64_encoded_float32_array...",
    "embedding_dim": 512,
    "model": "ArcFace",
    "enrollment_version": 1548
  }
}
```

**Edge → Broker (ACK):**
```json
{
  "type": "ack",
  "message_id": "enroll-update-12345",
  "timestamp": "2026-01-22T10:30:00.100Z",
  "payload": {
    "status": "success",
    "enrollment_version": 1548,
    "local_count": 2342
  }
}
```

---

### Enrollment Delete Message

Pushed when an enrollment is removed.

**Broker → Edge:**
```json
{
  "type": "enrollment_delete",
  "message_id": "enroll-delete-12346",
  "timestamp": "2026-01-22T10:30:00Z",
  "require_ack": true,
  "payload": {
    "employee_id": "EMP100001",
    "enrollment_version": 1549
  }
}
```

---

### Command Message

Device control commands.

**Broker → Edge:**
```json
{
  "type": "command",
  "message_id": "cmd-12347",
  "timestamp": "2026-01-22T10:30:00Z",
  "require_ack": true,
  "payload": {
    "command": "restart_recognition",
    "params": {
      "reason": "Model update"
    }
  }
}
```

**Supported Commands:**

| Command | Description | Parameters |
|---------|-------------|------------|
| `restart_recognition` | Restart recognition pipeline | `reason` (optional) |
| `reload_config` | Reload device configuration | None |
| `force_sync` | Trigger full enrollment sync | None |
| `capture_diagnostic` | Capture diagnostic frame | `include_video` (bool) |
| `set_log_level` | Change log verbosity | `level` (DEBUG/INFO/WARN/ERROR) |

---

## Edge to Archive Server API

### Upload Video Clip

Upload event-triggered video clip with metadata.

```http
POST /api/archive/clips
Host: archive.qraie.ai
Content-Type: multipart/form-data
X-Device-ID: cam-001
X-Tenant-ID: hbss
```

**Multipart Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | JSON | Clip metadata (see below) |
| `file` | Binary | Video file (H.265/H.264 MP4) |

**Metadata JSON:**
```json
{
  "clip_id": "clip-1737550200-cam001",
  "device_id": "cam-001",
  "tenant_id": "hbss",
  "event_id": "EV1-1737550200123-a1b2c3d4",
  "event_type": "face_recognition",
  "user_id": "EMP123456",
  "capture_time": "2026-01-22T10:30:00Z",
  "duration_seconds": 15,
  "pre_event_seconds": 10,
  "post_event_seconds": 5,
  "codec": "h265",
  "resolution": "1280x720",
  "fps": 25,
  "file_size_bytes": 2457600,
  "checksum_sha256": "abc123..."
}
```

**Response (Success - 201):**
```json
{
  "success": true,
  "clip_id": "clip-1737550200-cam001",
  "storage_path": "s3://archive/hbss/2026/01/22/cam-001/clip-1737550200-cam001.mp4",
  "retention_days": 90
}
```

---

### Upload Log Batch

Upload batched log entries.

```http
POST /api/archive/logs
Host: archive.qraie.ai
Content-Type: application/json
Content-Encoding: gzip
X-Device-ID: cam-001
X-Tenant-ID: hbss
```

**Request Body:**
```json
{
  "batch_id": "log-batch-1737550200-001",
  "device_id": "cam-001",
  "tenant_id": "hbss",
  "batch_info": {
    "count": 127,
    "first_timestamp": "2026-01-22T10:25:00.000Z",
    "last_timestamp": "2026-01-22T10:30:00.000Z",
    "sequence_start": 184500,
    "sequence_end": 184626,
    "compressed": true,
    "schema_version": "1.0"
  },
  "logs": [
    {
      "seq": 184500,
      "ts": "2026-01-22T10:25:00.123Z",
      "level": "INFO",
      "logger": "recognition",
      "message": "Face detected",
      "context": {
        "track_id": "face_0",
        "bbox": [100, 50, 200, 200],
        "frame_id": 12847
      }
    },
    {
      "seq": 184501,
      "ts": "2026-01-22T10:25:00.156Z",
      "level": "INFO",
      "logger": "recognition",
      "message": "Recognition match",
      "context": {
        "track_id": "face_0",
        "user_id": "EMP123456",
        "distance": 0.28,
        "confidence": 0.72
      }
    }
  ]
}
```

**Response (Success - 200):**
```json
{
  "success": true,
  "batch_id": "log-batch-1737550200-001",
  "accepted": 127,
  "last_sequence": 184626,
  "next_expected": 184627
}
```

---

## Recognition Validation Workflow

Events are only emitted after multi-frame validation to prevent false positives.

### Validation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOGNITION VALIDATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frame N     Frame N+1    Frame N+2    Frame N+3    Frame N+4  │
│     │            │            │            │            │       │
│     ▼            ▼            ▼            ▼            ▼       │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐      │
│  │Detect│────►│Detect│────►│Detect│────►│Detect│────►│Detect│   │
│  │Face │     │Face │     │Face │     │Face │     │Face │      │
│  └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘      │
│     │            │            │            │            │       │
│     ▼            ▼            ▼            ▼            ▼       │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐      │
│  │Match│     │Match│     │Match│     │Match│     │Match│      │
│  │EMP01│     │EMP01│     │EMP01│     │EMP01│     │EMP01│      │
│  │0.72 │     │0.75 │     │0.71 │     │0.73 │     │0.74 │      │
│  └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘      │
│     │            │            │            │            │       │
│     └────────────┴────────────┴────────────┴────────────┘       │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │ VALIDATE:        │                         │
│                    │ • 5 frames ✓     │                         │
│                    │ • 100% EMP01 ✓   │                         │
│                    │ • Avg conf 0.73  │                         │
│                    │ • Not in cooldown│                         │
│                    └────────┬─────────┘                         │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────┐                         │
│                    │  EMIT EVENT      │──────► IoT Broker       │
│                    │  + Start cooldown│                         │
│                    └──────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Validation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confirmation_frames` | 5 | Consecutive frames required |
| `consistency_threshold` | 0.80 | Minimum ratio of matching IDs |
| `distance_threshold` | 0.35 | Maximum cosine distance for valid match |
| `cooldown_seconds` | 30 | Seconds before same user triggers new event |
| `max_track_age_seconds` | 5.0 | Max age before track reset |

### State Machine

```
Detected ──► Tracking ──► Validated ──► Emitted ──► Cooldown
    │            │             │                        │
    │            │             │                        │
    └────────────┴─────────────┴────────────────────────┘
              (face lost or timeout)
```

---

## Video Buffer & Clip Capture

### Ring Buffer Design

A 15-second rolling buffer continuously stores video frames on SSD.

```
┌─────────────────────────────────────────────────────────────────┐
│                     15-SECOND RING BUFFER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Time:  T-15s    T-10s     T-5s      T(event)    T+5s          │
│           │        │         │           │          │           │
│           ▼        ▼         ▼           ▼          ▼           │
│        ┌──────────────────────────────────────────────┐         │
│        │ ████████████████████████░░░░░░░░░░░░░░░░░░░ │         │
│        │  Pre-event (10s)       │   Post-event (5s)  │         │
│        └──────────────────────────────────────────────┘         │
│                                 │                               │
│                                 ▼                               │
│                        Event Triggered                          │
│                                 │                               │
│                                 ▼                               │
│                    ┌──────────────────┐                         │
│                    │ Extract 15s clip │                         │
│                    │ Encode H.265     │                         │
│                    │ Queue for upload │                         │
│                    └──────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Buffer Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total duration | 15 seconds | Configurable |
| Pre-event | 10 seconds | Captures context before event |
| Post-event | 5 seconds | Captures reaction/confirmation |
| Resolution | 1280x720 | Configurable per device |
| Frame rate | 25 fps | Matches detection pipeline |
| Storage format | Raw YUV420 | Fast write, SSD-friendly |
| Storage location | `/var/qraie/video_buffer/` | SSD required |
| Buffer size | ~45 MB | At 720p 25fps |

### Clip Encoding

When event triggers clip extraction:

| Parameter | Value |
|-----------|-------|
| Codec | H.265 (NVENC hardware) |
| Container | MP4 |
| CRF | 28 (good quality, small size) |
| Preset | `medium` (balance speed/size) |
| Expected size | ~1.5-2.5 MB per 15s clip |

---

## Log Shipping Protocol

### Batching Strategy

Logs are batched locally before shipping:

| Trigger | Condition |
|---------|-----------|
| Count | 100 log entries |
| Time | 30 seconds elapsed |
| Size | 500 KB accumulated |
| Priority | ERROR/CRITICAL logs |

### Local Persistence

```
/var/qraie/log_buffer/
├── current.jsonl         # Active log file (append-only)
├── batch_001.jsonl.gz    # Completed batch (compressed, pending upload)
├── batch_002.jsonl.gz    # Completed batch (pending upload)
└── shipped/              # Successfully shipped (retained 24h)
    └── batch_000.jsonl.gz
```

### Log Entry Schema

```json
{
  "seq": 184500,
  "ts": "2026-01-22T10:25:00.123Z",
  "level": "INFO",
  "logger": "recognition",
  "message": "Recognition match",
  "context": {
    "track_id": "face_0",
    "user_id": "EMP123456",
    "distance": 0.28,
    "confidence": 0.72,
    "frame_id": 12847
  },
  "event_id": "EV1-1737550200123-a1b2c3d4",
  "duration_ms": 42
}
```

### Logger Names

| Logger | Subsystem |
|--------|-----------|
| `capture` | Video capture, frame acquisition |
| `detection` | Face detection |
| `recognition` | Face recognition/matching |
| `validator` | Event validation |
| `iot_client` | IoT broker communication |
| `sync` | Enrollment database sync |
| `buffer` | Video ring buffer |
| `archive` | Archive uploads |
| `system` | System metrics, health |

---

## Data Types Reference

### EventType

```
"face_recognition" | "emotion"
```

### EmotionCode

| Code | Emotion |
|------|---------|
| `EV2.1` | Neutral |
| `EV2.2` | Happiness |
| `EV2.3` | Anger |
| `EV2.4` | Contempt |
| `EV2.5` | Disgust |
| `EV2.6` | Fear |
| `EV2.7` | Sadness |
| `EV2.8` | Surprise |

### DeviceStatus

```
"operational" | "degraded" | "error" | "offline"
```

### LogLevel

```
"DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
```

### EventId Format

- Face recognition: `EV1-{timestamp_ms}-{random_hex8}`
- Emotion: `{emotion_code}-{timestamp_ms}-{random_hex8}`

Example: `EV1-1737550200123-a1b2c3d4`

---

## Error Handling & Retry Strategy

### HTTP Retry Policy

| Status Code | Action | Max Retries | Backoff |
|-------------|--------|-------------|---------|
| 429 | Retry with backoff | 5 | Exponential (1s, 2s, 4s, 8s, 16s) |
| 500, 502, 503, 504 | Retry with backoff | 3 | Exponential |
| 400, 401, 403, 404 | Do not retry | 0 | N/A |

### WebSocket Reconnection

| Event | Action | Delay |
|-------|--------|-------|
| Connection lost | Reconnect | 1s, then exponential backoff |
| Auth failure | Re-authenticate, reconnect | 5s |
| Server disconnect | Reconnect | 1s |
| Max retries exceeded | Alert, fallback to REST sync | N/A |

### Local Queue Behavior

When network unavailable:
1. Events queued locally (max 1000 events)
2. Clips queued to disk (max 100 clips or 1GB)
3. Logs queued to disk (max 10,000 entries or 50MB)
4. Automatic retry when connectivity restored
5. FIFO order preserved

---

## Status Codes & Errors

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `201` | Created |
| `207` | Multi-Status (partial success) |
| `400` | Bad Request |
| `401` | Unauthorized |
| `403` | Forbidden |
| `404` | Not Found |
| `409` | Conflict |
| `429` | Too Many Requests |
| `500` | Internal Server Error |
| `503` | Service Unavailable |

### Error Response Format

```json
{
  "success": false,
  "error": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional context"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `DEVICE_NOT_REGISTERED` | Device ID not found in registry |
| `INVALID_EVENT_FORMAT` | Event payload validation failed |
| `ENROLLMENT_NOT_FOUND` | Referenced user not enrolled |
| `SYNC_VERSION_MISMATCH` | Enrollment version conflict |
| `RATE_LIMITED` | Too many requests |
| `UPLOAD_TOO_LARGE` | File exceeds size limit |

---

## Jetson Deployment Guide

### Prerequisites

- Jetson Orin Nano with JetPack 5.x or 6.x
- SSD storage (for video buffer)
- PoE camera connected via Ethernet
- Internet connectivity (WiFi or secondary Ethernet)

### Directory Structure

```
/opt/qraie/
├── facial_recognition/
│   ├── main.py              # Main entry point
│   ├── video_buffer.py      # Video ring buffer
│   ├── iot_integration/     # IoT client module
│   └── venv/                # Python virtual environment
├── config/
│   └── device_config.json   # Device configuration
├── data/
│   ├── enrollments/         # Local enrollment database
│   │   └── enrollments.db
│   └── video_buffer/        # Video clip storage
└── logs/
    └── current.jsonl        # Local log buffer
```

### Quick Start

**1. Run headless setup script:**
```bash
# Download and run setup script
sudo ./jetson_edge/scripts/setup_jetson_headless.sh

# For full GUI removal (saves 3-5GB):
sudo ./jetson_edge/scripts/setup_jetson_headless.sh --full-removal
```

**2. Install the service:**
```bash
sudo ./jetson_edge/scripts/install_service.sh
```

**3. Configure the device:**
```bash
sudo nano /opt/qraie/config/device_config.json
```

Edit these fields:
- `device_id`: Unique identifier (e.g., "jetson-lobby-001")
- `camera.rtsp_url`: Your camera RTSP URL with credentials
- `broker_url`: IoT broker endpoint

**4. Start the service:**
```bash
sudo systemctl enable qraie-facial
sudo systemctl start qraie-facial
```

**5. Monitor:**
```bash
# Check status
sudo systemctl status qraie-facial

# View logs
sudo journalctl -u qraie-facial -f
```

### Configuration Reference

```json
{
  "device_id": "jetson-001",
  "broker_url": "https://acetaxi-bridge.qryde.net/iot-broker/api",
  
  "camera": {
    "rtsp_url": "rtsp://admin:PASSWORD@10.42.0.159/Preview_01_sub",
    "resolution": [1280, 720],
    "fps": 25
  },
  
  "recognition": {
    "model": "ArcFace",
    "detector_backend": "yolov8",
    "distance_threshold": 0.35,
    "min_face_size": 60
  },
  
  "validation": {
    "confirmation_frames": 5,
    "consistency_threshold": 0.80,
    "cooldown_seconds": 30
  },
  
  "video_buffer": {
    "enabled": true,
    "duration_seconds": 15,
    "pre_event_seconds": 10,
    "post_event_seconds": 5,
    "buffer_path": "/opt/qraie/data/video_buffer",
    "codec": "h265"
  },
  
  "heartbeat": {
    "interval_seconds": 30
  }
}
```

### Headless Mode

To run Jetson in CLI-only mode (recommended for production):

**Quick disable (keeps packages):**
```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

**Full removal (saves ~5GB):**
```bash
# Remove desktop packages
sudo apt-get purge ubuntu-desktop gnome-* gdm3

# IMPORTANT: Keep network-manager
sudo apt-get install network-manager

sudo systemctl set-default multi-user.target
sudo reboot
```

### Network Setup (PoE Camera)

For cameras on isolated PoE network:

```bash
# Share internet via Jetson ethernet
sudo nmcli con add type ethernet con-name "CameraNet" ifname eth0 ipv4.method shared
sudo nmcli con up CameraNet

# Enable IP forwarding
echo "net.ipv4.ip_forward=1" | sudo tee /etc/sysctl.d/99-forward.conf
sudo sysctl -p /etc/sysctl.d/99-forward.conf
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Service won't start | Check logs: `journalctl -u qraie-facial -n 100` |
| Camera connection failed | Verify RTSP URL: `ffprobe rtsp://...` |
| No GPU detected | Check `nvidia-smi`, ensure CUDA paths set |
| High CPU usage | Reduce FPS or resolution in config |
| Out of memory | Reduce `max_frames` in video buffer config |

### Service Management

```bash
# Start/Stop/Restart
sudo systemctl start qraie-facial
sudo systemctl stop qraie-facial
sudo systemctl restart qraie-facial

# Enable/Disable auto-start
sudo systemctl enable qraie-facial
sudo systemctl disable qraie-facial

# View status
sudo systemctl status qraie-facial

# View logs (live)
sudo journalctl -u qraie-facial -f

# View logs (last 100 lines)
sudo journalctl -u qraie-facial -n 100
```

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1 | 2026-01-22 | System | Added Jetson deployment guide |
| 1.0 | 2026-01-22 | System | Initial specification |
