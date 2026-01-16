# Facial Enrollment Modal - Schema Reference

## Type Definitions

### Enums / Union Types

```typescript
// User's enrollment status from backend
type EnrollmentStatus = 
  | 'unenrolled'              // No enrollment data exists
  | 'enrolled'                // All selected devices successfully synced
  | 'pending:embedding'       // Waiting on DeepFace vectorizer (no response)
  | 'pending:iot_confirmation'// Waiting for IoT broker response
  | 'pending:partial'         // IoT partial completion (not all devices synced)
  | 'pending:error';          // Error occurred, no "complete" from IoT

// Helper to check if status is any pending state
function isPending(status: EnrollmentStatus): boolean {
  return status.startsWith('pending:');
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
  user_id: string;
  device_list: string[];           // Device IDs targeted
  embedding: string;               // Base64 float32 array
  embedding_dim: number;           // 512 for ArcFace
  model: string;                   // "ArcFace"
  thumbnail: string;               // Base64 JPEG 128x128
  overview_sync_status: EnrollmentStatus;
  sync_details: DeviceSyncStatus[];
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
    embedding BLOB NOT NULL,             -- Float32 array (512 dims for ArcFace)
    embedding_normalized BLOB,           -- L2-normalized for cosine similarity
    image_count INTEGER,                 -- Number of successful captures
    profile_image_path TEXT,             -- Path to 128x128 thumbnail
    images_directory TEXT,               -- Path to captured images folder
    enrolled_at TEXT                     -- ISO timestamp
);
```
