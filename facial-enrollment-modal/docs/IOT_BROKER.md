# IoT Broker Integration Guide

This document covers the enrollment modal's communication with three systems:

1. **IoT Broker** - Device enumeration, sync status, embedding deployment
2. **DeepFace Vectorizer** - Embedding generation from captured images
3. **Dashboard Bridge** - Final enrollment data submission

---

## Multi-Tenant Architecture

The IoT Broker uses a multi-tenant architecture where tenant ID is extracted from the request Host header.

### Tenant ID Extraction

| Pattern | Example | Tenant ID | Database |
|---------|---------|-----------|----------|
| `{tenant-id}-bridge{env}.{domain}` | `hbss-bridgestg.qraie.ai` | `hbss` | `hbss-bridge` |
| `{tenant-id}-bridge{env}.{domain}` | `acetaxi-bridge.qraie.ai` | `acetaxi` | `acetaxi-bridge` |

### Required Headers

```typescript
// All API requests must include tenant identification
headers: {
  'Host': '{tenant-id}-bridgestg.qraie.ai'  // Primary method
}

// For service-to-service calls:
headers: {
  'X-Tenant-ID': '{tenant-id}'
}
```

---

## Microservice Architecture

The IoT Broker consists of three microservices:

| Service | Port | Base Path | Purpose |
|---------|------|-----------|---------|
| **Data Service** | 3001 | `/api/data/*` | Devices, Events, Enrollment CRUD |
| **Config Service** | 3002 | `/api/config/*` | Lookups, System configuration |
| **Broker Service** | 3000 | `/api/devices/*` | WebSocket communication, Device commands |

### Swagger Documentation

- Data Service: `http://localhost:3001/api-docs`
- Config Service: `http://localhost:3002/api-docs`
- Broker Service: `http://localhost:3000/api-docs`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ENROLLMENT MODAL                                   │
│                                                                                 │
│   Part 1          Part 2              Part 3              Part 4                │
│   ┌─────┐         ┌─────┐             ┌─────┐             ┌─────┐               │
│   │Open │ ──────▶ │Cap- │ ──────────▶ │Push │ ──────────▶ │Return│              │
│   │Modal│         │ture │             │to   │             │to    │              │
│   └──┬──┘         └──┬──┘             │IoT  │             │Bridge│              │
│      │               │                └──┬──┘             └──┬───┘              │
└──────┼───────────────┼───────────────────┼───────────────────┼──────────────────┘
       │               │                   │                   │
       ▼               ▼                   ▼                   ▼
┌─────────────┐  ┌───────────────┐  ┌─────────────┐    ┌─────────────────┐
│ IoT Broker  │  │ DeepFace      │  │ IoT Broker  │    │ Dashboard       │
│ Data Svc    │  │ Vectorizer    │  │ Broker Svc  │    │ Bridge          │
│             │  │               │  │             │    │                 │
│ GET devices │  │ POST images   │  │ POST publish│    │ employee_id     │
│             │  │ user_id       │  │ employee_id │    │ embedded_file   │
│             │  │               │  │             │    │ thumbnail       │
│ Returns:    │  │ Returns:      │  │ WebSocket   │    │                 │
│ - devices   │  │ - embedding   │  │ broadcast   │    │                 │
│ - status    │  │ - thumbnail   │  │ to devices  │    │                 │
└─────────────┘  └───────────────┘  └─────────────┘    └─────────────────┘
```

---

## Enrollment Status States

### Overview Status (for Bridge API)

| Status | Description |
|--------|-------------|
| `unenrolled` | No enrollment data exists |
| `captured` | Images captured, `enrollmentProcessedFile` generated, awaiting publish |
| `published` | Successfully published to IoT edge devices |

### Detailed Status (Internal Logging)

| Status | Description |
|--------|-------------|
| `pending:embedding` | Waiting on DeepFace vectorizer (no response) |
| `pending:iot_confirmation` | Waiting for IoT broker response |
| `pending:partial` | IoT partial completion (not all devices synced) |
| `pending:error` | Error occurred, no "complete" from IoT |

### TypeScript Types

```typescript
// Simplified status for Bridge API
type EnrollmentStatus = 'unenrolled' | 'captured' | 'published';

// Detailed status for internal logging
type EnrollmentStatusDetailed = 
  | 'unenrolled'
  | 'captured'
  | 'published'
  | 'pending:embedding'
  | 'pending:iot_confirmation'
  | 'pending:partial'
  | 'pending:error';

interface DeviceSyncStatus {
  device_id: string;
  display_name: string;
  status: 'synced' | 'pending' | 'failed';
  last_sync_at?: string;
  error?: string;
}

interface EnrollmentSyncOverview {
  employee_id: string;
  enrollmentStatus: EnrollmentStatus;
  devices_targeted: number;
  devices_synced: number;
  devices_pending: number;
  devices_failed: number;
  device_details: DeviceSyncStatus[];
}
```

---

## 4-Part Enrollment Flow

### Part 1: Modal Opened (Device Enumeration)

**Flow:** `[IoT Broker Data Service] → [Modal]`

When the modal opens, it fetches the device list from the IoT Broker Data Service.

```
Modal Opens
    │
    ▼
GET /api/data/devices?device-type=face_recognition
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Device Deployment                            │
│─────────────────────────────────────────────────────────────────│
│  User: John Doe (EMP123456)                                     │
│  Current Status: pending:partial                                │
│                                                                 │
│  Enrollment will be published to all face recognition devices:  │
│                                                                 │
│  🟢 cam-001  Lobby Camera           Online                      │
│  🟢 cam-002  Break Room             Online                      │
│  🔴 cam-003  Parking (Offline)      Offline                     │
│  🟡 dev-001  New Kiosk              Provisioning                │
│                                                                 │
│  ───────────────────────────────────────────────────           │
│  [Cancel]  [Start Enrollment]                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### API: Get Device List

**GET `/api/data/devices`** - Returns available devices

**Query Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `device-type` | `string` | No | Filter: `"face_recognition"` or `"emotion_monitoring"`. Returns all devices if not provided. |

**Request Headers:**
```
Host: {tenant-id}-bridgestg.qraie.ai
```

**Response:**
```json
{
  "devices": [
    {
      "device_id": "cam-001",
      "display_name": "Lobby Camera",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Main entrance outside",
      "status": "online",
      "last_heartbeat_at": "2026-01-16T10:30:00Z"
    },
    {
      "device_id": "cam-002", 
      "display_name": "Break Room",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Building A - Floor 2",
      "status": "online",
      "last_heartbeat_at": "2026-01-16T10:29:52Z"
    },
    {
      "device_id": "cam-003",
      "display_name": "Parking",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Parking Structure B",
      "status": "offline",
      "last_heartbeat_at": "2026-01-16T08:15:00Z"
    }
  ],
  "total": 3,
  "online": 2,
  "offline": 1
}
```

---

### Part 2: Enrollment Process (Capture & Vectorize)

**Flow:** `[Modal] → [DeepFace Vectorizer] → [Modal]`

User captures 5 poses, images sent to DeepFace for embedding generation.

```
Capture Complete (5 images)
    │
    ▼
POST /api/vectorizer/generate
{
  "user_id": "EMP123456",
  "images": [
    { "pose": "front", "data": "base64..." },
    { "pose": "left", "data": "base64..." },
    ...
  ]
}
    │
    ▼
Status: pending:embedding (waiting for response)
    │
    ▼
Response from Vectorizer
{
  "employee_id": "EMP123456",
  "enrollmentProcessedFile": "base64_float32_array...",
  "embedding_dim": 512,
  "model": "ArcFace",
  "enrollmentPictureThumbnail": "base64_jpeg_128x128...",
  "image_count": 5
}
```

**If vectorizer fails/times out:** Status = `pending:embedding`

---

### Part 3: Enrollment Push (Deploy to IoT)

**Flow:** `[Modal] → [IoT Broker Data Service] → [Broker Service] → [Devices via WebSocket]`

Publish enrollment to all face recognition devices via the IoT Broker.

> **Important Change:** The new API simplifies enrollment publishing. You only need to provide `employee_id`. The system automatically retrieves the `embedded_file` from the database and broadcasts to ALL face recognition devices via WebSocket.

```
Embedding Stored in Database
    │
    ▼
POST /api/data/enrollment/publish
{
  "employee_id": "EMP123456"
}
    │
    ▼
Status: pending:iot_confirmation (waiting for response)
    │
    ▼
System automatically:
1. Retrieves embedded_file from database
2. Sends to ALL face_recognition devices via WebSocket
    │
    ▼
Response from IoT Broker
{
  "success": true,
  "message": "Enrollment published to all devices",
  "data": {
    "employee_id": "EMP123456",
    "devices_notified": 2
  }
}
```

**Status Outcomes:**
- All devices notified → `published`
- No response from IoT → `pending:iot_confirmation` (internal), `captured` (Bridge API)
- Error response → `pending:error` (internal), `captured` (Bridge API)

---

### Part 4: Return to Bridge

**Flow:** `[Modal] → [Dashboard Bridge]`

Modal sends complete enrollment data back to the Dashboard Bridge.

```
IoT Publish Complete
    │
    ▼
POST /api/bridge/enrollment/complete
{
  "employee_id": "EMP123456",
  "enrollmentProcessedFile": "base64_float32_array...",
  "embedding_dim": 512,
  "model": "ArcFace",
  "enrollmentPictureThumbnail": "base64_jpeg_128x128...",
  "enrollmentStatus": "published"
}
    │
    ▼
Bridge updates Employee Profile
Modal closes
onEnrollmentComplete callback fired
```

**Bridge Receives:**
| Field | Type | Description |
|-------|------|-------------|
| `employee_id` | `string` | Employee ID |
| `enrollmentProcessedFile` | `string` | Facial landmark embedded file (base64) |
| `enrollmentPictureThumbnail` | `string` | Base64 JPEG 128x128 |
| `enrollmentStatus` | `string` | `"captured"` or `"published"` |

---

## Part 1: Dashboard Bridge API

### Data Flow: Modal → Dashboard

After successful capture, the modal submits to the Dashboard Bridge, which stores:

| Field | Type | Description |
|-------|------|-------------|
| `employee_id` | `string` | WFM employee ID |
| `enrollmentStatus` | `'unenrolled'` \| `'captured'` \| `'published'` | Current state |
| `enrollmentPictureThumbnail` | `string` | Base64 JPEG (128x128 profile image) |
| `enrollmentProcessedFile` | `string` | Facial landmark embedded file for person identification |
| `embedding_model` | `string` | Model name (e.g., `"ArcFace"`) |
| `enrolled_at` | `ISO8601` | Enrollment timestamp |

### POST `/api/enrollment/capture`

**Request:**
```json
{
  "userId": "EMP123456",
  "captures": [
    { "pose": "front", "imageData": "data:image/jpeg;base64,..." },
    { "pose": "left", "imageData": "data:image/jpeg;base64,..." },
    { "pose": "right", "imageData": "data:image/jpeg;base64,..." },
    { "pose": "up", "imageData": "data:image/jpeg;base64,..." },
    { "pose": "down", "imageData": "data:image/jpeg;base64,..." }
  ]
}
```

**Response (stored in Employee Profile DB):**
```json
{
  "success": true,
  "message": "Enrollment complete",
  "data": {
    "employee_id": "EMP123456",
    "enrollmentStatus": "captured",
    "enrollmentPictureThumbnail": "data:image/jpeg;base64,/9j/4AAQ...",
    "enrollmentProcessedFile": "AACAPwAAgD8AAIA/...",
    "embedding_model": "ArcFace",
    "embedding_dim": 512,
    "image_count": 5,
    "enrolled_at": "2026-01-16T10:30:00Z"
  }
}
```

### Status Values

| Status | Meaning |
|--------|---------|
| `unenrolled` | No facial data exists |
| `captured` | Embedding generated, NOT yet pushed to devices |
| `published` | Embedding pushed to edge devices |

---

## Part 2: IoT Broker API

### Device Discovery

#### GET `/api/data/devices`

Returns list of edge devices. This is an **Enrollment System API**.

> **Note:** Aligns with `IoT_Devices` MongoDB collection schema from `IOT_BROKER_FRAMEWORK_revised.md`

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `device-type` | `string` | - | Filter: `"face_recognition"` or `"emotion_monitoring"`. Returns all devices if omitted. |

**Required Headers:**
```
Host: {tenant-id}-bridgestg.qraie.ai
```

**Response:**
```json
{
  "devices": [
    {
      "device_id": "cam-001",
      "display_name": "Lobby Entrance Camera",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Main entrance outside",
      "status": "online",
      "last_heartbeat_at": "2026-01-16T10:29:45Z"
    },
    {
      "device_id": "cam-002",
      "display_name": "Break Room Camera",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Building A - Floor 2",
      "status": "online",
      "last_heartbeat_at": "2026-01-16T10:29:52Z"
    },
    {
      "device_id": "cam-003",
      "display_name": "Parking Garage Entry",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Parking Structure B",
      "status": "offline",
      "last_heartbeat_at": "2026-01-16T08:15:00Z"
    }
  ],
  "total": 3,
  "online": 2,
  "offline": 1
}
```

### TypeScript Interface (matches IoT_Devices schema)

```typescript
/**
 * Maps to IoT_Devices MongoDB collection
 * See: docs/IOT_BROKER_FRAMEWORK_revised.md
 */
interface EdgeDevice {
  device_id: string;                              // Unique ID (e.g., "cam-001")
  display_name: string;                           // Human-readable name
  device_category: string;                        // "camera", etc.
  capability: 'face_recognition' | 'emotion_monitoring';
  location_label?: string;                        // Friendly location name
  status: 'online' | 'offline' | 'provisioning';
  last_heartbeat_at: string | null;               // ISO8601 or null
}

interface DeviceListResponse {
  devices: EdgeDevice[];
  total: number;
  online: number;
  offline: number;
}
```

---

### Publish Enrollment to Devices

#### POST `/api/data/enrollment/publish`

Publish enrollment data to all face recognition devices. This is an **Enrollment System API**.

> **Simplified API:** Only requires `employee_id`. The system automatically retrieves the `embedded_file` (facial landmark data) from the database and broadcasts to all face recognition devices via WebSocket.

**Request:**
```json
{
  "employee_id": "EMP123456"
}
```

**Required Headers:**
```
Host: {tenant-id}-bridgestg.qraie.ai
Content-Type: application/json
```

**What Gets Sent to Devices (via WebSocket):**
- `employee_id` (from request)
- `embedded_file` (automatically retrieved from database)
- `person_id` (from enrollment record)
- `person_name` (from enrollment record, if available)

**Response (Success):**
```json
{
  "success": true,
  "message": "Enrollment published to all devices",
  "data": {
    "employee_id": "EMP123456",
    "devices_notified": 2
  }
}
```

**Response (Error - Missing embedded_file):**
```json
{
  "success": false,
  "error": "Employee enrollment not found or embedded_file missing"
}
```

---

### Device Heartbeat

#### POST `/api/data/devices/{deviceId}/heartbeat`

Update the last heartbeat timestamp for a device.

**Path Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `deviceId` | `string` | Unique device identifier |

**Response:**
```json
{
  "success": true,
  "message": "Heartbeat updated",
  "last_heartbeat_at": "2026-01-16T10:30:00Z"
}
```

---

### Device Commands (WebSocket)

#### POST `/api/devices/{deviceId}/enrollment/publish`

Send enrollment data to a specific device via WebSocket. This is called internally by the Data Service.

**Request:**
```json
{
  "employee_id": "EMP123456",
  "person_id": "EMP123456",
  "person_name": "John Doe",
  "embedded_file": "base64_facial_landmark_data..."
}
```

**Response:**
```json
{
  "success": true,
  "message": "Enrollment publish command sent"
}
```

**Error (Device Not Connected):**
```json
{
  "success": false,
  "error": "Device not connected"
}
```

---

## Deprecated APIs

The following APIs from previous versions are **deprecated**:

| Deprecated Endpoint | Replacement | Notes |
|---------------------|-------------|-------|
| `GET /api/iot/devices` | `GET /api/data/devices` | Path changed |
| `POST /api/iot/enrollments/push` | `POST /api/data/enrollment/publish` | Simplified request body |
| `GET /api/iot/enrollments/sync` | WebSocket push | Devices now receive enrollments via WebSocket push |
| `GET /api/iot/enrollments/status/{user_id}` | N/A | Per-user sync status removed |

### Removed Parameters

The following parameters are no longer supported in the new API:

- `target_devices` - Enrollments now broadcast to ALL face_recognition devices
- `push_mode` - No longer needed (always broadcasts to all)
- `capability` query param - Renamed to `device-type`

---

## Complete Enrollment Flow

```
1. User clicks "Enroll Face" on Employee Card
                    │
                    ▼
2. EnrollmentModal opens, checks enrollmentStatus
                    │
        ┌───────────┴───────────┐
        │                       │
   'unenrolled'          'captured'/'published'
        │                       │
        │                       ▼
        │              Re-enrollment Confirmation
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
3. Capture 5 poses (front, left, right, up, down)
                    │
                    ▼
4. Submit to GPU Vectorizer
   POST /api/vectorizer/generate
                    │
                    ▼
5. Receive: enrollmentProcessedFile, enrollmentPictureThumbnail
   Status set to: 'captured'
                    │
                    ▼
6. Fetch device list from IoT Broker (optional, for display)
   GET /api/data/devices?device-type=face_recognition
                    │
                    ▼
7. Publish to IoT Broker
   POST /api/data/enrollment/publish
   { "employee_id": "..." }
                    │
                    ▼
8. System broadcasts to all devices via WebSocket
                    │
                    ▼
9. Update status to: 'published'
                    │
                    ▼
10. Modal closes, onEnrollmentComplete callback fired
```

---

## Security Considerations

### Authentication

- Dashboard Bridge uses session cookies / JWT from WFM auth
- IoT Broker extracts tenant from Host header
- For service-to-service calls, use `X-Tenant-ID` header

### Data Flow

- Embeddings are generated on the **server** (Python/DeepFace), never in browser
- Only base64 images are sent from browser to server
- Edge devices receive `embedded_file` via WebSocket, **not** original images

### Device Authorization

- Each edge device has unique `device_id`
- Devices connect to Broker Service via WebSocket
- Enrollment publish operations require valid tenant context

---

## Pydantic Schemas (Python Backend)

> Aligns with `IoT_Devices` and `IoT_Events` MongoDB schemas from `IOT_BROKER_FRAMEWORK_revised.md`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

# Enrollment status values
EnrollmentStatus = Literal["unenrolled", "captured", "published"]


class EdgeDevice(BaseModel):
    """
    Maps to IoT_Devices MongoDB collection.
    See: docs/IOT_BROKER_FRAMEWORK_revised.md
    """
    device_id: str = Field(..., description="Unique device ID (e.g., 'cam-001')")
    display_name: str = Field(..., description="Human-readable device name")
    device_category: str = Field(default="camera", description="Device type")
    capability: Literal["face_recognition", "emotion_monitoring"]
    location_label: Optional[str] = Field(None, description="Friendly location name")
    status: Literal["online", "offline", "provisioning"] = "provisioning"
    last_heartbeat_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PublishEnrollmentRequest(BaseModel):
    """
    Request to publish enrollment to all face recognition devices.
    The enrollmentProcessedFile is automatically retrieved from the database.
    """
    employee_id: str = Field(..., description="Employee/Person ID to publish enrollment for")


class PublishEnrollmentResponse(BaseModel):
    """Response from enrollment publish operation."""
    success: bool
    message: str
    data: Optional[dict] = None  # Contains employee_id, devices_notified
    error: Optional[str] = None


class DeviceEnrollmentPayload(BaseModel):
    """
    Payload sent to devices via WebSocket.
    This is constructed internally by the system.
    """
    employee_id: str
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    enrollmentProcessedFile: str = Field(..., description="Facial landmark embedded file (base64)")
    enrollment_data: Optional[dict] = None


class EmployeeEnrollment(BaseModel):
    """
    Enrollment data stored in Bridge employee collection.
    """
    employee_id: str
    enrollmentStatus: EnrollmentStatus = "unenrolled"
    enrollmentProcessedFile: Optional[str] = Field(None, description="Base64 Float32Array (512 dims)")
    enrollmentPictureThumbnail: Optional[str] = Field(None, description="Base64 JPEG 128x128")
    embedding_model: Optional[str] = Field(None, description="Model name (e.g., 'ArcFace')")
    embedding_dim: Optional[int] = Field(None, description="Embedding dimension (512 for ArcFace)")
    enrolled_at: Optional[datetime] = None


# Event schemas (for reference - see iot_integration/schemas/event_schemas.py)
class FaceRecognitionEvent(BaseModel):
    """
    Maps to IoT_Events MongoDB collection (event_type="face_recognition").
    Event ID format: EV1-{timestamp}-{random}
    """
    event_id: str = Field(..., description="Format: EV1-{timestamp}-{random}")
    device_id: str = Field(..., description="Reference to IoT_Devices.device_id")
    event_type: Literal["face_recognition"] = "face_recognition"
    event_timestamp: datetime
    person_name: Optional[str] = None
    person_id: Optional[str] = None
    metadata: Optional[dict] = None
    debug: List[dict] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
```
