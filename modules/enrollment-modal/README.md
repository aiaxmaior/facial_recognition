# Enrollment Modal

Facial enrollment interface for the QRAIE system. Runs on the **Bridge server** and provides webcam-based face capture for employee enrollment.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Bridge Server                      │
│  ┌───────────────────────────────────────────────┐  │
│  │           Enrollment Modal                     │  │
│  │                                                │  │
│  │  ┌─────────────┐      ┌─────────────────────┐ │  │
│  │  │   React     │ ───▶ │   Node.js API       │ │  │
│  │  │   Frontend  │      │   Server            │ │  │
│  │  └─────────────┘      └──────────┬──────────┘ │  │
│  └──────────────────────────────────┼────────────┘  │
└─────────────────────────────────────┼───────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                                   ▼
          ┌─────────────────┐               ┌─────────────────┐
          │   GPU Server    │               │   IoT Broker    │
          │   (Vectorizer)  │               │   (Publish)     │
          └─────────────────┘               └─────────────────┘
```

## Components

| Directory | Description |
|-----------|-------------|
| `packages/api-server/` | Node.js backend API |
| `packages/react-component/` | React enrollment UI |
| `python-api/` | Python API wrapper |
| `audio/` | Audio guidance files |

## API Endpoints

### Internal (Modal API)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/enrollment/capture` | POST | Submit captured images |
| `/api/enrollment/publish/:id` | POST | Publish to edge devices |
| `/api/enrollment/status/:id` | GET | Get enrollment status |

### Outbound Calls

| Target | Endpoint | Purpose |
|--------|----------|---------|
| GPU Server | `POST /api/vectorizer/generate` | Generate facial embedding |
| IoT Broker | `POST /api/data/enrollment/publish` | Broadcast to edge devices |

## Data Formats

| Data | Format | Description |
|------|--------|-------------|
| Embedding | Base64 Float32Array | 512-dimension vector |
| Thumbnail | Base64 JPEG | 128x128 pixels |
| Captures | Base64 JPEG | 5 poses (front, left, right, up, down) |

## Installation

```bash
cd modules/enrollment-modal
npm install
```

## Configuration

Create `.env` file:

```env
GPU_SERVER_URL=https://gpu-server.example.com/api
IOT_BROKER_URL=https://acetaxi-bridge.qryde.net/iot-broker/api
REDIS_URL=redis://localhost:6379
PORT=3000
```

## Running

```bash
# Development
npm run dev

# Production
npm run build
npm start
```

## Enrollment Flow

1. User clicks "Enroll Face" in WFM Dashboard
2. Modal captures 5 pose images from webcam
3. Images sent to GPU Server → returns embedding + thumbnail
4. Enrollment published to IoT Broker → broadcasts to edge devices
5. Enrollment stored in Bridge database
