# IoT Broker Integration Guide

This document covers the two distinct integration points for the enrollment modal:

1. **Dashboard Bridge** - Stores enrollment data in central WFM system
2. **IoT Broker** - Pushes facial profiles to edge devices

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            WFM Dashboard                                    │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │ Employee Card   │───▶│ EnrollmentModal  │───▶│ Dashboard Bridge API   │  │
│  │ (React)         │    │ (React)          │    │ /api/enrollment/*      │  │
│  └─────────────────┘    └──────────────────┘    └────────────────────────┘  │
│                                                           │                  │
│                                                           ▼                  │
│                                               ┌────────────────────────┐    │
│                                               │ Employee Profile DB    │    │
│                                               │ (embedding, status,    │    │
│                                               │  thumbnail)            │    │
│                                               └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                                           │ Publish
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IoT Broker                                     │
│  ┌────────────────────────┐         ┌─────────────────────────────────────┐ │
│  │ Device Registry        │         │ Enrollment Sync Service             │ │
│  │ GET /devices           │         │ POST /enrollments/push              │ │
│  │ - List edge devices    │         │ - Push profiles to selected devices │ │
│  │ - Filter by capability │         │ - Track sync status per device      │ │
│  └────────────────────────┘         └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                          ┌────────────────┼────────────────┐
                                          │                │                │
                                          ▼                ▼                ▼
                                   ┌───────────┐    ┌───────────┐    ┌───────────┐
                                   │ Edge      │    │ Edge      │    │ Edge      │
                                   │ Device 1  │    │ Device 2  │    │ Device N  │
                                   │ (Jetson)  │    │ (Jetson)  │    │ (Jetson)  │
                                   └───────────┘    └───────────┘    └───────────┘
```

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

Returns list of edge devices with facial recognition capability.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `capability` | `string` | `"facial_recognition"` | Filter by device capability |
| `status` | `string` | `"online"` | Filter by device status |
| `location` | `string` | - | Filter by location/site |

**Response:**
```json
{
  "devices": [
    {
      "device_id": "cam-lobby-001",
      "name": "Lobby Entrance Camera",
      "location": "Building A - Main Lobby",
      "status": "online",
      "capabilities": ["facial_recognition", "emotion_detection"],
      "last_heartbeat": "2026-01-16T10:29:45Z",
      "enrolled_count": 142,
      "sync_version": 847
    },
    {
      "device_id": "cam-breakroom-002",
      "name": "Break Room Camera",
      "location": "Building A - Floor 2",
      "status": "online",
      "capabilities": ["facial_recognition"],
      "last_heartbeat": "2026-01-16T10:29:52Z",
      "enrolled_count": 142,
      "sync_version": 847
    },
    {
      "device_id": "cam-parking-003",
      "name": "Parking Garage Entry",
      "location": "Parking Structure B",
      "status": "offline",
      "capabilities": ["facial_recognition"],
      "last_heartbeat": "2026-01-16T08:15:00Z",
      "enrolled_count": 140,
      "sync_version": 845
    }
  ],
  "total": 3,
  "online": 2,
  "offline": 1
}
```

### TypeScript Interface

```typescript
interface EdgeDevice {
  device_id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'maintenance';
  capabilities: ('facial_recognition' | 'emotion_detection')[];
  last_heartbeat: string;  // ISO8601
  enrolled_count: number;
  sync_version: number;
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
  "target_devices": ["cam-lobby-001", "cam-breakroom-002"],
  "push_mode": "selected"
}
```

**Push Modes:**
| Mode | Description |
|------|-------------|
| `"selected"` | Push only to devices in `target_devices` array |
| `"all"` | Push to ALL online devices with facial_recognition capability |
| `"location"` | Push to all devices at a specific location |

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
        "device_id": "cam-lobby-001",
        "success": true,
        "sync_version": 848
      },
      {
        "device_id": "cam-breakroom-002",
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
      { "device_id": "cam-lobby-001", "success": true },
      { "device_id": "cam-breakroom-002", "success": true },
      { "device_id": "cam-parking-003", "success": false, "error": "Device offline" }
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
│  ☑️  🟢 Lobby Entrance Camera                                   │
│        Building A - Main Lobby                                  │
│                                                                 │
│  ☑️  🟢 Break Room Camera                                       │
│        Building A - Floor 2                                     │
│                                                                 │
│  ☐  🔴 Parking Garage Entry (Offline)                          │
│        Parking Structure B                                      │
│                                                                 │
│  ───────────────────────────────────────────────────           │
│  ☐  Select All Online Devices (2)                              │
│                                                                 │
│           [Cancel]    [Deploy to Selected (2)]                  │
└─────────────────────────────────────────────────────────────────┘
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

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class EdgeDevice(BaseModel):
    device_id: str
    name: str
    location: str
    status: str  # 'online', 'offline', 'maintenance'
    capabilities: List[str]
    last_heartbeat: datetime
    enrolled_count: int
    sync_version: int

class PushRequest(BaseModel):
    employee_id: str
    target_devices: List[str] = []
    push_mode: str = "selected"  # 'selected', 'all', 'location'
    location: Optional[str] = None

class PushResult(BaseModel):
    device_id: str
    success: bool
    sync_version: Optional[int] = None
    error: Optional[str] = None

class PushResponse(BaseModel):
    success: bool
    message: str
    data: dict  # Contains employee_id, devices_updated, results, etc.
```
