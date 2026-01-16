# Dashboard Integration Guide

## Quick Start

### 1. Install the Package

```bash
npm install @qryde/facial-enrollment
# or
yarn add @qryde/facial-enrollment
```

### 2. Basic Integration

```tsx
import { EnrollmentModal } from '@qryde/facial-enrollment';
import { useState } from 'react';

function EmployeeCard({ employee }) {
  const [showEnrollment, setShowEnrollment] = useState(false);

  return (
    <div className="employee-card">
      <span>{employee.name}</span>
      <button onClick={() => setShowEnrollment(true)}>
        {employee.enrollmentStatus === 'enrolled' ? 'Re-enroll' : 'Enroll Face'}
      </button>

      <EnrollmentModal
        isOpen={showEnrollment}
        onClose={() => setShowEnrollment(false)}
        userId={employee.id}
        userName={employee.name}
        enrollmentStatus={employee.enrollmentStatus}
        apiEndpoint="/api/enrollment"
        onEnrollmentComplete={(result) => {
          console.log('Enrolled:', result);
          // Refresh employee data
        }}
      />
    </div>
  );
}
```

---

## Component Props

### EnrollmentModalProps

| Prop | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `isOpen` | `boolean` | ✅ | - | Controls modal visibility |
| `onClose` | `() => void` | ✅ | - | Called when modal should close |
| `userId` | `string` | ✅ | - | Employee/user ID for enrollment |
| `enrollmentStatus` | `EnrollmentStatus` | ✅ | - | Current status: `'enrolled'`, `'pending'`, or `'unenrolled'` |
| `apiEndpoint` | `string` | ✅ | - | Base URL for enrollment API |
| `onEnrollmentComplete` | `(result) => void` | ❌ | - | Success callback with enrollment result |
| `userName` | `string` | ❌ | - | Display name (shown in re-enrollment prompt) |
| `enableAudio` | `boolean` | ❌ | `true` | Enable/disable audio guidance |
| `className` | `string` | ❌ | - | Additional CSS class for styling |

---

## API Endpoints Required

Your backend must implement these endpoints:

### GET `/api/enrollment/status/:userId`
Get enrollment status for a user.

**Response:**
```json
{
  "userId": "EMP123",
  "status": "enrolled",
  "enrolledAt": "2026-01-15T10:30:00Z",
  "imageCount": 5,
  "profileImageUrl": "/images/EMP123/profile_128.jpg"
}
```

### POST `/api/enrollment/capture`
Submit captured images for embedding generation.

**Request:**
```json
{
  "userId": "EMP123",
  "captures": [
    { "pose": "front", "imageData": "data:image/jpeg;base64,/9j/4AAQ..." },
    { "pose": "left", "imageData": "data:image/jpeg;base64,/9j/4AAQ..." },
    { "pose": "right", "imageData": "data:image/jpeg;base64,/9j/4AAQ..." },
    { "pose": "up", "imageData": "data:image/jpeg;base64,/9j/4AAQ..." },
    { "pose": "down", "imageData": "data:image/jpeg;base64,/9j/4AAQ..." }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully enrolled with 5 images",
  "data": {
    "userId": "EMP123",
    "embeddingCount": 5,
    "profileImagePath": "/images/EMP123/profile_128.jpg",
    "status": "pending"
  }
}
```

### POST `/api/enrollment/publish/:userId`
Push embeddings to IoT edge devices (optional).

**Response:**
```json
{
  "success": true,
  "message": "Published to 3 devices",
  "data": {
    "userId": "EMP123",
    "status": "enrolled",
    "devicesUpdated": 3
  }
}
```

### DELETE `/api/enrollment/:userId`
Remove enrollment data.

**Response:**
```json
{
  "success": true,
  "message": "Enrollment deleted"
}
```

---

## Re-enrollment Flow

When `enrollmentStatus` is `'enrolled'` or `'pending'`, the modal automatically shows a confirmation dialog:

```
┌─────────────────────────────────────────────────┐
│           ⚠️ Existing Profile Found             │
│                                                 │
│  [userName] already has a facial recognition    │
│  profile that has been published to devices.    │
│                                                 │
│  Would you like to recreate the facial          │
│  profile for this employee?                     │
│                                                 │
│  This will replace the existing biometric data. │
│                                                 │
│      [Cancel]    [Yes, Recreate Profile]        │
└─────────────────────────────────────────────────┘
```

- **Cancel**: Closes modal, no changes
- **Yes, Recreate**: Proceeds to capture flow, will overwrite existing data

---

## Audio Files

The component expects audio files at `/audio/` (configurable via API server):

**Primary (required):**
- `look_forward.mp3` - "Look straight at the camera"
- `turn_left.mp3` - "Turn your head left"  
- `turn_right.mp3` - "Turn your head right"
- `look_up.mp3` - "Tilt your chin up"
- `look_down.mp3` - "Tilt your chin down"
- `beep.mp3` - Countdown beep
- `capture_complete.mp3` - Final capture sound

**Secondary guidance (optional):**
- `guidance_left_more.mp3`, `guidance_left_exceeded.mp3`
- `guidance_right_more.mp3`, `guidance_right_exceeded.mp3`
- `guidance_up_more.mp3`, `guidance_up_exceeded.mp3`
- `guidance_down_more.mp3`, `guidance_down_exceeded.mp3`
- `guidance_level_chin.mp3`, `guidance_face_forward.mp3`

---

## State Flow Diagram

```
                    ┌──────────────────────────────────────┐
                    │           Modal Opened               │
                    └──────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
           enrollmentStatus              enrollmentStatus
           = 'unenrolled'                = 'enrolled'/'pending'
                    │                                 │
                    ▼                                 ▼
            ┌───────────┐                   ┌─────────────────┐
            │   IDLE    │                   │ Re-enroll       │
            │  (Start   │                   │ Confirmation    │
            │  Button)  │                   └─────────────────┘
            └───────────┘                            │
                    │                    ┌───────────┴───────────┐
                    │                    │                       │
                    │               [Cancel]              [Confirm]
                    │                    │                       │
                    │                    ▼                       │
                    │              Close Modal                   │
                    │                                            │
                    └──────────────────┬─────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   CAPTURING     │
                              │  (5 poses)      │
                              │                 │
                              │ front → left →  │
                              │ right → up →    │
                              │ down            │
                              └─────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   PROCESSING    │
                              │  (API call)     │
                              └─────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                       Success                    Error
                          │                         │
                          ▼                         ▼
                    ┌───────────┐            ┌───────────┐
                    │ COMPLETE  │            │  ERROR    │
                    │ (10s auto │            │ (Retry)   │
                    │  close)   │            └───────────┘
                    └───────────┘
```

---

## Styling / Theming

The modal uses inline styles but accepts a `className` prop for customization:

```tsx
<EnrollmentModal
  className="my-custom-modal"
  // ...other props
/>
```

```css
/* Override modal styles */
.my-custom-modal .enrollment-modal-overlay {
  /* Custom overlay */
}
```

---

## Browser Requirements

- **Camera**: `getUserMedia` API (HTTPS required in production)
- **WebGL**: Required for TensorFlow.js face detection
- **ES2020+**: Modern JavaScript features

**Tested browsers:**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## Development Setup

```bash
# Clone and install
cd facial-enrollment-modal
npm install

# Run demo (http://localhost:5173)
npm run dev

# Build component
npm run build:component
```

### Architecture

```
facial-enrollment-modal/
├── packages/
│   ├── react-component/    # Main exportable React component
│   │   ├── src/
│   │   │   ├── components/ # UI components
│   │   │   ├── hooks/      # useCamera, useFacePose, useEnrollment
│   │   │   ├── services/   # API client, audio player
│   │   │   ├── types/      # TypeScript interfaces
│   │   │   └── constants.ts
│   │   └── package.json
│   └── api-server/         # Node.js proxy to Python backend
├── demo/                   # Vite demo application
└── python-api/             # Flask wrapper for DeepFace
```

---

## Troubleshooting

### "Camera initialization timeout"
- Ensure HTTPS (or localhost)
- Check browser camera permissions
- Try different browser

### "Face detection not working"
- Face detection model takes 10-20s to load on first use
- Ensure good lighting
- Face should be clearly visible in frame

### "No audio"
- Check browser autoplay policies
- User must interact with page before audio can play
- Verify audio files are served at correct path

### "Re-enrollment not showing confirmation"
- Verify `enrollmentStatus` prop is `'enrolled'` or `'pending'`
- Check that status is passed correctly from backend
