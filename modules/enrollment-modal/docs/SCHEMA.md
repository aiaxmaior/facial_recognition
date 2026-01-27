# Facial Enrollment Modal - Schema Reference

## Type Definitions

### Enums / Union Types

```typescript
// User's enrollment status from backend (simplified for Bridge API)
type EnrollmentStatus = 'unenrolled' | 'captured' | 'published';

// Detailed internal status for logging/debugging
type EnrollmentStatusDetailed = 
  | 'unenrolled'              // No enrollment data exists
  | 'captured'                // Images captured, embedding generated
  | 'published'               // Successfully published to IoT devices
  | 'pending:embedding'       // Waiting on DeepFace vectorizer (no response)
  | 'pending:iot_confirmation'// Waiting for IoT broker response
  | 'pending:partial'         // IoT partial completion (not all devices synced)
  | 'pending:error';          // Error occurred, no "complete" from IoT

// Helper to check if status is any pending state
function isPending(status: EnrollmentStatusDetailed): boolean {
  return status.startsWith('pending:');
}

// Map detailed status to simplified status for Bridge API
function toSimplifiedStatus(status: EnrollmentStatusDetailed): EnrollmentStatus {
  if (status === 'unenrolled') return 'unenrolled';
  if (status === 'published') return 'published';
  return 'captured'; // All pending states map to 'captured'
}

// Internal state machine for capture flow
type CaptureState = 
  | 'idle'          // Waiting to start
  | 'initializing'  // Loading camera/models
  | 'capturing'     // Actively capturing poses
  | 'processing'    // Sending to backend for embedding generation
  | 'complete'      // Successfully enrolled
  | 'error';        // Something went wrong

// The 5 capture poses (order matters)
type PoseName = 'front' | 'left' | 'right' | 'up' | 'down';
```

---

## Multi-Tenant Configuration

### TenantConfig
Configuration for multi-tenant IoT Broker communication.

```typescript
interface TenantConfig {
  /** Tenant identifier extracted from host (e.g., "hbss", "acetaxi") */
  tenantId: string;
  
  /** Full tenant host for API requests (e.g., "hbss-bridgestg.qraie.ai") */
  tenantHost: string;
  
  /** Environment suffix (e.g., "stg" for staging, "" for production) */
  environment: 'stg' | 'prod' | '';
  
  /** Base domain (e.g., "qraie.ai") */
  domain: string;
}

/**
 * Extract tenant configuration from a host string
 * Pattern: {tenant-id}-bridge{env}.{domain}
 */
function parseTenantHost(host: string): TenantConfig {
  // Example: "hbss-bridgestg.qraie.ai" -> { tenantId: "hbss", environment: "stg", ... }
  const match = host.match(/^([a-z0-9]+)-bridge(stg|prod)?\.(.+)$/i);
  if (!match) throw new Error(`Invalid tenant host: ${host}`);
  
  return {
    tenantId: match[1],
    tenantHost: host,
    environment: (match[2] || '') as TenantConfig['environment'],
    domain: match[3]
  };
}
```

### API Headers
Required headers for multi-tenant API calls.

```typescript
interface TenantHeaders {
  /** Primary method: Host header with tenant subdomain */
  Host: string;
}

interface ServiceToServiceHeaders {
  /** For internal service-to-service calls */
  'X-Tenant-ID': string;
}

// Usage example
const headers: TenantHeaders = {
  Host: 'hbss-bridgestg.qraie.ai'
};
```

---

## Core Interfaces

### PoseTarget
Defines requirements for each capture pose.

```typescript
interface PoseTarget {
  name: PoseName;
  displayName: string;         // "Front View", "Left View", etc.
  instruction: string;         // User-facing instruction text
  yawRange: [number, number];  // [min, max] degrees for horizontal rotation
  pitchRange: [number, number]; // [min, max] degrees for vertical rotation
  audioFile: string;           // Audio prompt filename
}
```

### FaceLandmark
Single point from MediaPipe's 468-landmark face mesh.

```typescript
interface FaceLandmark {
  x: number;  // Normalized 0-1 (horizontal position)
  y: number;  // Normalized 0-1 (vertical position)
  z: number;  // Depth (relative, for 3D pose estimation)
}
```

### FacePose
Real-time face detection result from TensorFlow.js.

```typescript
interface FacePose {
  detected: boolean;
  yaw: number;      // Left/right rotation (-90 to +90 degrees)
  pitch: number;    // Up/down rotation (-90 to +90 degrees)
  roll: number;     // Head tilt (-90 to +90 degrees)
  confidence: number;
  landmarks?: FaceLandmark[];  // 468 points for mesh visualization
}
```

### CapturedFrame
Data for a single captured image.

```typescript
interface CapturedFrame {
  pose: PoseName;
  imageData: string;  // Base64 encoded JPEG (data:image/jpeg;base64,...)
  timestamp: number;  // Unix timestamp (ms)
}
```

### EnrollmentResult
Backend response after successful enrollment.

```typescript
interface EnrollmentResult {
  success: boolean;
  userId: string;
  message: string;
  embeddingCount?: number;     // Number of valid face embeddings generated
  profileImageUrl?: string;    // URL to 128x128 profile thumbnail
}
```

### DeviceSyncStatus
Per-device sync status for IoT deployment.

```typescript
interface DeviceSyncStatus {
  device_id: string;
  display_name: string;
  device_status: 'online' | 'offline' | 'provisioning';
  sync_status: 'synced' | 'pending' | 'failed' | 'not_deployed';
  last_sync_at?: string;       // ISO8601 timestamp
  error?: string;              // Error message if failed
}
```

### EnrollmentSyncOverview
Complete sync overview returned to Dashboard Bridge.

```typescript
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

### BridgeSubmission
Final data sent to Dashboard Bridge after enrollment.

```typescript
interface BridgeSubmission {
  /** Employee ID */
  employee_id: string;
  /** Facial landmark embedded file for person identification (base64 Float32Array) */
  enrollmentProcessedFile: string;
  /** Embedding dimension (512 for ArcFace) */
  embedding_dim: number;
  /** Model name (e.g., "ArcFace") */
  model: string;
  /** Base64 JPEG 128x128 profile thumbnail */
  enrollmentPictureThumbnail: string;
  /** Final enrollment status: "captured" or "published" */
  enrollmentStatus: EnrollmentStatus;
}
```

> **Note:** The `device_list` and `sync_details` fields have been removed. 
> The new API broadcasts to all face recognition devices automatically.

---

## IoT Broker API Types

### EdgeDevice
Device information from IoT Broker Data Service.

```typescript
interface EdgeDevice {
  device_id: string;
  display_name: string;
  device_category: string;
  capability: 'face_recognition' | 'emotion_monitoring';
  location_label?: string;
  status: 'online' | 'offline' | 'provisioning';
  last_heartbeat_at: string | null;
}

interface DeviceListResponse {
  devices: EdgeDevice[];
  total: number;
  online: number;
  offline: number;
}
```

### PublishEnrollmentRequest
Request to publish enrollment to IoT devices.

> **Simplified API:** Only requires `employee_id`. The system automatically retrieves 
> the `embedded_file` from the database and broadcasts to all face recognition devices.

```typescript
interface PublishEnrollmentRequest {
  /** Employee/Person ID to publish enrollment for */
  employee_id: string;
}
```

### PublishEnrollmentResponse
Response from enrollment publish operation.

```typescript
interface PublishEnrollmentResponse {
  success: boolean;
  message: string;
  data?: {
    employee_id: string;
    devices_notified: number;
  };
  error?: string;
}
```

### Deprecated Types

The following types are **deprecated** and should not be used:

```typescript
/** @deprecated Use PublishEnrollmentRequest instead */
interface PushRequest_DEPRECATED {
  employee_id: string;
  target_devices: string[];  // REMOVED - now broadcasts to all devices
  push_mode: 'selected' | 'all' | 'location';  // REMOVED
  location_label?: string;  // REMOVED
}
```

---

## API Request/Response Types

### SubmitCapturesRequest
POST body for `/api/enrollment/capture`

```typescript
interface SubmitCapturesRequest {
  userId: string;
  captures: Array<{
    pose: PoseName;
    imageData: string;  // Base64 JPEG
  }>;
}
```

### EnrollmentApiResponse
Response from enrollment endpoints.

```typescript
interface EnrollmentApiResponse {
  success: boolean;
  message: string;
  data?: {
    userId: string;
    embeddingCount: number;
    profileImagePath: string;
    status: EnrollmentStatus;
  };
  error?: string;
}
```

### StatusApiResponse
Response from `/api/enrollment/status/:userId`

```typescript
interface StatusApiResponse {
  userId: string;
  status: EnrollmentStatus;
  enrolledAt?: string;      // ISO timestamp
  imageCount?: number;
  profileImageUrl?: string;
}
```

---

## Constants

### Capture Targets (matches Python backend)

| Pose | Yaw Range | Pitch Range | Audio File |
|------|-----------|-------------|------------|
| `front` | -25° to 25° | -20° to 20° | `look_forward.mp3` |
| `left` | 10° to 50° | -30° to 30° | `turn_left.mp3` |
| `right` | -50° to -10° | -30° to 30° | `turn_right.mp3` |
| `up` | -30° to 30° | 15° to 50° | `look_up.mp3` |
| `down` | -40° to 40° | -50° to -15° | `look_down.mp3` |

### Timing Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `COUNTDOWN_SECONDS` | 3 | Seconds before auto-capture |
| `POSE_HOLD_TIME` | 0.5s | Hold time before countdown starts |
| `STABLE_FRAME_THRESHOLD` | 10 | Consecutive valid frames required |
| `COUNTDOWN_TOLERANCE` | 40° | Extra tolerance during countdown |
| `AUTO_CLOSE_DELAY` | 10s | Auto-close after completion |
| `ADVISORY_AUDIO_INTERVAL` | 3.0s | Gap between guidance audio |
| `ADVISORY_AUDIO_MAX` | 5 | Max guidance prompts per pose |

### API Endpoints

| Service | Port | Base Path |
|---------|------|-----------|
| Data Service | 3001 | `/api/data/*` |
| Config Service | 3002 | `/api/config/*` |
| Broker Service | 3000 | `/api/devices/*` |

---

## Database Schema (SQLite - faces.db)

```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,           -- Display name (First_Last)
    first_name TEXT,
    last_name TEXT,
    employee_id TEXT,                    -- External employee ID
    email TEXT,
    model TEXT NOT NULL,                 -- "ArcFace"
    detector TEXT,                       -- "yolov8"
    enrollmentProcessedFile BLOB NOT NULL,    -- Facial landmark embedded file (512 dims for ArcFace)
    enrollmentProcessedFile_normalized BLOB,  -- L2-normalized for cosine similarity
    image_count INTEGER,                 -- Number of successful captures
    enrollmentPictureThumbnail TEXT,     -- Path to 128x128 thumbnail
    images_directory TEXT,               -- Path to captured images folder
    enrollmentStatus TEXT,               -- "unenrolled", "captured", or "published"
    enrolled_at TEXT                     -- ISO timestamp
);
```

## Bridge Employee Collection Schema

Fields to be added to the existing employee collection:

```javascript
{
  // ... existing employee fields ...
  
  // Enrollment fields (added)
  enrollmentStatus: String,              // "unenrolled" | "captured" | "published"
  enrollmentProcessedFile: Binary,       // Facial landmark embedding (512-dim Float32Array)
  enrollmentPictureThumbnail: String,    // Base64 JPEG 128x128 or URL path
  enrolled_at: Date                      // Timestamp of last enrollment
}
```
