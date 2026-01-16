# IoT Broker Integration Guide

This document covers the enrollment modal's communication with three systems:

1. **IoT Broker** - Device enumeration, sync status, embedding deployment
2. **DeepFace Vectorizer** - Embedding generation from captured images
3. **Dashboard Bridge** - Final enrollment data submission

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
│             │  │ Vectorizer    │  │             │    │ Bridge          │
│ GET devices │  │               │  │ POST push   │    │                 │
│ GET sync    │  │ POST images   │  │ user_id     │    │ user_id         │
│ status      │  │ user_id       │  │ embedding   │    │ device_list     │
│             │  │               │  │ device_list │    │ embedding       │
│ Returns:    │  │ Returns:      │  │             │    │ thumbnail       │
│ - devices   │  │ - embedding   │  │ Returns:    │    │ sync_status     │
│ - sync_stat │  │ - thumbnail   │  │ - user_id   │    │                 │
└─────────────┘  └───────────────┘  │ - sync_stat │    └─────────────────┘
                                    │ - devices   │
                                    └─────────────┘
```

---

## Enrollment Status States

### Overview Status (for Bridge)

| Status | Description |
|--------|-------------|
| `unenrolled` | No enrollment data exists |
| `enrolled` | All selected devices successfully synced |
| `pending:embedding` | Waiting on DeepFace vectorizer (no response) |
| `pending:iot_confirmation` | Waiting for IoT broker response |
| `pending:partial` | IoT partial completion (not all devices synced) |
| `pending:error` | Error occurred, no "complete" from IoT |

### TypeScript Types

```typescript
type EnrollmentStatus = 
  | 'unenrolled'
  | 'enrolled'
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
  user_id: string;
  overview_status: EnrollmentStatus;
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

**Flow:** `[IoT Broker] → [Modal]`

When the modal opens, it immediately fetches device list and sync status from the IoT Broker.

```
Modal Opens
    │
    ▼
GET /api/iot/devices?capability=face_recognition
GET /api/iot/enrollments/status/{user_id}
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Device Deployment                            │
│─────────────────────────────────────────────────────────────────│
│  User: John Doe (EMP123456)                                     │
│  Current Status: pending:partial                                │
│                                                                 │
│  Select devices for facial profile deployment:                  │
│                                                                 │
│  ☑️  🟢 cam-001  Lobby Camera           ✅ Synced              │
│  ☑️  🟢 cam-002  Break Room             ⏳ Pending             │
│  ☐  🔴 cam-003  Parking (Offline)       ❌ Failed              │
│  ☐  🟡 dev-001  New Kiosk               ⚪ Not deployed        │
│                                                                 │
│  ───────────────────────────────────────────────────           │
│  [Cancel]  [Start Enrollment]  [Re-enroll All Selected]        │
└─────────────────────────────────────────────────────────────────┘
```

#### API: Get Device List with Sync Status

**GET `/api/iot/devices`** - Returns available devices

**GET `/api/iot/enrollments/status/{user_id}`** - Returns sync status per device

**Response:**
```json
{
  "user_id": "EMP123456",
  "overview_status": "pending:partial",
  "devices": [
    {
      "device_id": "cam-001",
      "display_name": "Lobby Camera",
      "device_status": "online",
      "sync_status": "synced",
      "last_sync_at": "2026-01-16T10:30:00Z"
    },
    {
      "device_id": "cam-002", 
      "display_name": "Break Room",
      "device_status": "online",
      "sync_status": "pending",
      "last_sync_at": null
    },
    {
      "device_id": "cam-003",
      "display_name": "Parking",
      "device_status": "offline",
      "sync_status": "failed",
      "error": "Device unreachable"
    }
  ]
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
  "user_id": "EMP123456",
  "embedding": "base64_float32_array...",
  "embedding_dim": 512,
  "model": "ArcFace",
  "thumbnail": "base64_jpeg_128x128...",
  "image_count": 5
}
```

**If vectorizer fails/times out:** Status = `pending:embedding`

---

### Part 3: Enrollment Push (Deploy to IoT)

**Flow:** `[Modal] → [IoT Broker] → [Modal]`

Push embedding to selected edge devices via IoT Broker.

```
Embedding Received
    │
    ▼
POST /api/iot/enrollments/push
{
  "user_id": "EMP123456",
  "embedding": "base64_float32_array...",
  "embedding_dim": 512,
  "model": "ArcFace",
  "target_devices": ["cam-001", "cam-002"]
}
    │
    ▼
Status: pending:iot_confirmation (waiting for response)
    │
    ▼
Response from IoT Broker
{
  "user_id": "EMP123456",
  "sync_results": [
    { "device_id": "cam-001", "status": "synced" },
    { "device_id": "cam-002", "status": "synced" }
  ],
  "overview_status": "enrolled"
}
```

**Status Outcomes:**
- All devices synced → `enrolled`
- Some devices failed → `pending:partial`
- No response from IoT → `pending:iot_confirmation`
- Error response → `pending:error`

---

### Part 4: Return to Bridge

**Flow:** `[Modal] → [Dashboard Bridge]`

Modal sends complete enrollment data back to the Dashboard Bridge.

```
IoT Sync Complete
    │
    ▼
POST /api/bridge/enrollment/complete
{
  "user_id": "EMP123456",
  "device_list": ["cam-001", "cam-002"],
  "embedding": "base64_float32_array...",
  "embedding_dim": 512,
  "model": "ArcFace",
  "thumbnail": "base64_jpeg_128x128...",
  "overview_sync_status": "enrolled",
  "sync_details": [
    { "device_id": "cam-001", "status": "synced" },
    { "device_id": "cam-002", "status": "synced" }
  ]
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
| `user_id` | `string` | Employee ID |
| `device_list` | `string[]` | Devices targeted for deployment |
| `embedding` | `string` | Base64 float32 array (512 dims) |
| `thumbnail` | `string` | Base64 JPEG 128x128 |
| `overview_sync_status` | `EnrollmentStatus` | Final status |
| `sync_details` | `DeviceSyncStatus[]` | Per-device results |

---

## Part 1: Dashboard Bridge API

### Data Flow: Modal → Dashboard

After successful capture, the modal submits to the Dashboard Bridge, which stores:

| Field | Type | Description |
|-------|------|-------------|
| `employee_id` | `string` | WFM employee ID |
| `enrollment_status` | `'enrolled'` \| `'pending'` \| `'unenrolled'` | Current state |
| `thumbnail` | `string` | Base64 JPEG (128x128 profile image) |
| `embedding` | `string` | Base64 float32 array (512 dims for ArcFace) |
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
    "enrollment_status": "pending",
    "thumbnail": "data:image/jpeg;base64,/9j/4AAQ...",
    "embedding": "AACAPwAAgD8AAIA/...",
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
| `pending` | Embedding generated, NOT pushed to devices |
| `enrolled` | Embedding pushed to one or more edge devices |

---

## Part 2: IoT Broker API

### Device Discovery

#### GET `/api/iot/devices`

Returns list of edge devices with face recognition capability.

> **Note:** Aligns with `IoT_Devices` MongoDB collection schema from `IOT_BROKER_FRAMEWORK_revised.md`

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `capability` | `string` | `"face_recognition"` | Filter: `"face_recognition"` or `"emotion_monitoring"` |
| `status` | `string` | `"online"` | Filter: `"online"`, `"offline"`, or `"provisioning"` |
| `location_label` | `string` | - | Filter by location label |

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
      "location_geo": [-122.4194, 37.7749],
      "status": "online",
      "last_heartbeat_at": "2026-01-16T10:29:45Z",
      "enrolled_count": 142,
      "sync_version": 847
    },
    {
      "device_id": "cam-002",
      "display_name": "Break Room Camera",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Building A - Floor 2",
      "location_geo": null,
      "status": "online",
      "last_heartbeat_at": "2026-01-16T10:29:52Z",
      "enrolled_count": 142,
      "sync_version": 847
    },
    {
      "device_id": "cam-003",
      "display_name": "Parking Garage Entry",
      "device_category": "camera",
      "capability": "face_recognition",
      "location_label": "Parking Structure B",
      "location_geo": [-122.4180, 37.7755],
      "status": "offline",
      "last_heartbeat_at": "2026-01-16T08:15:00Z",
      "enrolled_count": 140,
      "sync_version": 845
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
  location_geo?: [number, number] | null;         // [longitude, latitude]
  status: 'online' | 'offline' | 'provisioning';
  last_heartbeat_at: string | null;               // ISO8601 or null
  
  // Extended fields for enrollment sync (not in base schema)
  enrolled_count?: number;
  sync_version?: number;
}

interface DeviceListResponse {
  devices: EdgeDevice[];
  total: number;
  online: number;
  offline: number;
}
```

---

### Push Enrollment to Devices

#### POST `/api/iot/enrollments/push`

Push a facial profile to selected edge devices.

**Request:**
```json
{
  "employee_id": "EMP123456",
  "target_devices": ["cam-001", "cam-002"],
  "push_mode": "selected"
}
```

**Push Modes:**
| Mode | Description |
|------|-------------|
| `"selected"` | Push only to devices in `target_devices` array |
| `"all"` | Push to ALL online devices with `capability="face_recognition"` |
| `"location"` | Push to all devices matching `location_label` |

**Response:**
```json
{
  "success": true,
  "message": "Profile pushed to 2 devices",
  "data": {
    "employee_id": "EMP123456",
    "devices_targeted": 2,
    "devices_updated": 2,
    "devices_failed": 0,
    "new_status": "enrolled",
    "results": [
      {
        "device_id": "cam-001",
        "success": true,
        "sync_version": 848
      },
      {
        "device_id": "cam-002",
        "success": true,
        "sync_version": 848
      }
    ]
  }
}
```

**Error Response (partial failure):**
```json
{
  "success": false,
  "message": "Profile push partially failed",
  "data": {
    "employee_id": "EMP123456",
    "devices_targeted": 3,
    "devices_updated": 2,
    "devices_failed": 1,
    "new_status": "pending",
    "results": [
      { "device_id": "cam-001", "success": true },
      { "device_id": "cam-002", "success": true },
      { "device_id": "cam-003", "success": false, "error": "Device offline" }
    ]
  }
}
```

---

### Enrollment Sync Schema (Edge Device Pull)

Edge devices pull enrollment updates from the broker:

#### GET `/api/iot/enrollments/sync`

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `device_id` | `string` | Device requesting sync |
| `since_version` | `int` | Last known sync version (0 for full sync) |
| `model` | `string` | Expected model (e.g., `"ArcFace"`) |

**Response (Incremental Sync):**
```json
{
  "sync_version": 848,
  "additions": [
    {
      "user_id": "EMP123456",
      "embedding": "AACAPwAAgD8AAIA/...",
      "dim": 512,
      "model": "ArcFace",
      "display_name": "John Doe"
    }
  ],
  "removals": ["EMP999888"],
  "full_sync_required": false,
  "total_enrolled": 143,
  "synced_at": "2026-01-16T10:30:15Z"
}
```

---

## UI Flow: Device Selection

The enrollment modal should include a device deployment step after capture:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Deploy to Edge Devices                        │
│─────────────────────────────────────────────────────────────────│
│                                                                 │
│  Select devices to receive this facial profile:                │
│                                                                 │
│  ☑️  🟢 Lobby Entrance Camera          [cam-001]               │
│        📍 Main entrance outside                                 │
│                                                                 │
│  ☑️  🟢 Break Room Camera              [cam-002]               │
│        📍 Building A - Floor 2                                  │
│                                                                 │
│  ☐  🔴 Parking Garage Entry (Offline)  [cam-003]               │
│        📍 Parking Structure B                                   │
│                                                                 │
│  ☐  🟡 New Kiosk (Provisioning)        [dev-001]               │
│        📍 Reception Desk                                        │
│                                                                 │
│  ───────────────────────────────────────────────────           │
│  ☐  Select All Online Devices (2)                              │
│                                                                 │
│           [Cancel]    [Deploy to Selected (2)]                  │
└─────────────────────────────────────────────────────────────────┘

Legend: 🟢 online  🔴 offline  🟡 provisioning
```

### Suggested Component Props Extension

```typescript
interface EnrollmentModalProps {
  // ... existing props ...
  
  /** Enable device deployment step after capture */
  enableDeviceDeployment?: boolean;
  
  /** IoT Broker API endpoint (separate from dashboard API) */
  iotBrokerEndpoint?: string;
  
  /** Callback when deployment completes */
  onDeploymentComplete?: (result: DeploymentResult) => void;
  
  /** Pre-select specific devices */
  defaultTargetDevices?: string[];
  
  /** Auto-deploy to all devices (skip selection) */
  autoDeployAll?: boolean;
}

interface DeploymentResult {
  employeeId: string;
  devicesTargeted: number;
  devicesUpdated: number;
  devicesFailed: number;
  newStatus: EnrollmentStatus;
}
```

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
   'unenrolled'          'enrolled'/'pending'
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
4. Submit to Dashboard Bridge API
   POST /api/enrollment/capture
                    │
                    ▼
5. Receive: employee_id, embedding, thumbnail
   Status set to: 'pending'
                    │
                    ▼
6. [If enableDeviceDeployment=true]
   Fetch device list from IoT Broker
   GET /api/iot/devices?capability=facial_recognition
                    │
                    ▼
7. User selects target devices
                    │
                    ▼
8. Push to IoT Broker
   POST /api/iot/enrollments/push
   { employee_id, target_devices }
                    │
                    ▼
9. Update status to: 'enrolled'
                    │
                    ▼
10. Modal closes, onEnrollmentComplete callback fired
```

---

## Security Considerations

### Authentication
- Dashboard Bridge uses session cookies / JWT from WFM auth
- IoT Broker requires separate API key (`X-API-Key` header)

### Data Flow
- Embeddings are generated on the **server** (Python/DeepFace), never in browser
- Only base64 images are sent from browser to server
- Edge devices receive embeddings, **not** original images

### Device Authorization
- Each edge device has unique `device_id` and API credentials
- Broker validates device_id on all requests
- Push operations require admin-level WFM permissions

---

## Pydantic Schemas (Python Backend)

> Aligns with `IoT_Devices` and `IoT_Events` MongoDB schemas from `IOT_BROKER_FRAMEWORK_revised.md`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

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
    location_geo: Optional[List[float]] = Field(None, description="[longitude, latitude]")
    status: Literal["online", "offline", "provisioning"] = "provisioning"
    last_heartbeat_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Extended fields for enrollment sync
    enrolled_count: Optional[int] = None
    sync_version: Optional[int] = None


class PushRequest(BaseModel):
    """Request to push enrollment to edge devices."""
    employee_id: str
    target_devices: List[str] = []
    push_mode: Literal["selected", "all", "location"] = "selected"
    location_label: Optional[str] = None  # For push_mode="location"


class PushResult(BaseModel):
    """Result for a single device push."""
    device_id: str
    success: bool
    sync_version: Optional[int] = None
    error: Optional[str] = None


class PushResponse(BaseModel):
    """Response from enrollment push operation."""
    success: bool
    message: str
    data: dict  # Contains employee_id, devices_updated, results, etc.


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
    metadata: Optional[dict] = None  # { person_name: str, confidence: float, ... }
    debug: List[dict] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
```
