# Facial Enrollment Modal

A React/TypeScript facial enrollment system that integrates with existing WFM dashboards as a popup modal.

## Features

- **5-pose face capture**: Front, Left, Right, Up, Down
- **Real-time face pose detection** using TensorFlow.js (MediaPipe FaceMesh)
- **Audio guidance** for accessibility
- **Automatic capture** with countdown when pose is correct
- **Progress tracking** with thumbnail previews
- **Backend integration** with existing Python/DeepFace system

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (React)                          │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ EnrollmentModal │  │ TensorFlow.js   │                   │
│  │   Component     │──│ Face Detection  │                   │
│  └────────┬────────┘  └─────────────────┘                   │
│           │                                                 │
└───────────┼─────────────────────────────────────────────────┘
            │ HTTP
┌───────────▼─────────────────────────────────────────────────┐
│                  Node.js API Server                         │
│  - /api/enrollment/capture                                  │
│  - /api/enrollment/status/:userId                           │
│  - /api/enrollment/publish/:userId                          │
└───────────┬─────────────────────────────────────────────────┘
            │ HTTP
┌───────────▼─────────────────────────────────────────────────┐
│              Python DeepFace Backend                        │
│  - Face embedding generation (ArcFace)                      │
│  - SQLite database storage                                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Node.js 18+ 
- Python 3.8+ (for DeepFace backend, optional for demo)
- npm or yarn

### Installation

```bash
cd facial-enrollment-modal

# Install all dependencies
npm install
```

### Running the Demo

You need to run three services:

#### Terminal 1: Node.js API Server
```bash
npm run dev:api
```
This starts the API server on http://localhost:3001

#### Terminal 2: React Demo App
```bash
npm run dev:demo
```
This starts the demo dashboard on http://localhost:5173

#### Terminal 3: Python Backend (Optional)
```bash
cd python-api
pip install -r requirements.txt
python enrollment_api.py
```
This starts the DeepFace processing on http://localhost:5000

> **Note**: The demo works without the Python backend - it will use mock processing. 
> The Python backend is only needed for actual face embedding generation.

### All-in-One (Without Python)
```bash
npm run dev
```
This starts both the API server and demo app concurrently.

## Integration Guide

### Installing the Component

```bash
npm install @qryde/facial-enrollment
```

### Using in Your React App

```tsx
import { EnrollmentModal } from '@qryde/facial-enrollment';

function MyComponent() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button onClick={() => setIsOpen(true)}>
        Enroll Biometrics
      </button>

      <EnrollmentModal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        userId="employee-123"
        enrollmentStatus="unenrolled" // 'enrolled' | 'pending' | 'unenrolled'
        apiEndpoint="/api/enrollment"
        onEnrollmentComplete={(result) => {
          console.log('Enrolled:', result);
        }}
        enableAudio={true}
      />
    </>
  );
}
```

### Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `isOpen` | `boolean` | Yes | Whether the modal is visible |
| `onClose` | `() => void` | Yes | Called when modal should close |
| `userId` | `string` | Yes | User ID for enrollment |
| `enrollmentStatus` | `'enrolled' \| 'pending' \| 'unenrolled'` | Yes | Current status |
| `apiEndpoint` | `string` | Yes | API base URL |
| `onEnrollmentComplete` | `(result) => void` | No | Called on successful enrollment |
| `userName` | `string` | No | Display name for user |
| `enableAudio` | `boolean` | No | Enable audio guidance (default: true) |
| `className` | `string` | No | Additional CSS class |

## Project Structure

```
facial-enrollment-modal/
├── packages/
│   ├── react-component/    # Main React component library
│   │   ├── src/
│   │   │   ├── components/ # UI components
│   │   │   ├── hooks/      # React hooks
│   │   │   ├── services/   # API & audio services
│   │   │   └── types/      # TypeScript types
│   │   └── package.json
│   │
│   └── api-server/         # Node.js API server
│       └── src/
│           ├── routes/     # Express routes
│           └── services/   # Backend services
│
├── python-api/             # Python DeepFace wrapper
│   ├── enrollment_api.py
│   └── requirements.txt
│
├── demo/                   # Demo React app
│   └── src/
│       └── App.tsx         # Mock employee dashboard
│
└── package.json            # Monorepo root
```

## Enrollment States

1. **Unenrolled**: User has no biometric data
2. **Pending**: Embeddings generated but not published to IoT devices
3. **Enrolled**: Fully enrolled and synced to edge devices

## API Endpoints

### GET /api/enrollment/status/:userId
Get enrollment status for a user.

### POST /api/enrollment/capture
Submit captured images for processing.

```json
{
  "userId": "string",
  "captures": [
    { "pose": "front", "imageData": "base64..." },
    { "pose": "left", "imageData": "base64..." },
    ...
  ]
}
```

### POST /api/enrollment/publish/:userId
Publish enrollment to IoT edge devices.

### DELETE /api/enrollment/:userId
Delete enrollment data.

## Audio Files

Place audio files in the `audio/` directory:
- `look_forward.mp3`
- `turn_left.mp3`
- `turn_right.mp3`
- `look_up.mp3`
- `look_down.mp3`
- `hold_pose.mp3`
- `beep.mp3`
- `capture_complete.mp3`
- `guidance_*.mp3` (optional secondary prompts)

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 14+
- Edge 80+

Requires:
- WebRTC (getUserMedia)
- WebGL (for TensorFlow.js)

## License

MIT
