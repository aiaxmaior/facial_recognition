# Data Portal

Web interface for visualizing and reviewing pipeline results.

## Overview

The portal provides:
- Interactive data tables for clips and videos
- Analysis visualizations (emotions, actions, demographics)
- Video preview player
- API access to all analysis data

## Quick Start

```bash
cd /media/ajax/AI/Diffusion/training_data/portal
./start.sh
```

**Access:**
- Portal UI: http://localhost:3000
- API Docs: http://localhost:8088/docs

## Architecture

```
portal/
├── backend/
│   ├── main.py           # FastAPI server
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx       # Main application
│   │   └── components/   # React components
│   ├── package.json
│   └── vite.config.ts
├── start.sh              # Launch script
└── README.md
```

## API Endpoints

### Main Pipeline Data
| Endpoint | Description |
|----------|-------------|
| `GET /api/clips` | All clip data with analyses |
| `GET /api/videos` | Video-level aggregations |
| `GET /api/insights` | Summary statistics |
| `GET /api/pipeline-data` | Combined pipeline data |

### HAR Data
| Endpoint | Description |
|----------|-------------|
| `GET /api/har-actions` | Full HAR action data |
| `GET /api/har-pipeline-data` | HAR data for portal |

### Video Access
| Endpoint | Description |
|----------|-------------|
| `GET /video/{path}` | Stream video file |
| `GET /thumbnail/{path}` | Get video thumbnail |

## Frontend Components

### CaptionReviewer
Review and edit VLM-generated captions.

### ClipsTable
Interactive table with:
- Sorting by any column
- Filtering by video, emotion, action
- Click to view details

### VideosTable
Aggregated video statistics:
- Clip counts
- Action distribution
- Duration totals

### InsightsPanel
Dashboard showing:
- Total clips/videos
- Emotion distribution
- Action distribution
- Processing status

## Data Sources

The portal reads from `analysis/`:
- `detections.json` - Person detection
- `emotions.json` - Emotion analysis
- `demographics.json` - Age/gender
- `nudenet.json` - Content classification
- `har_actions.json` - Action recognition
- `captions.json` - VLM descriptions

## Development

### Backend
```bash
cd portal/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8088
```

### Frontend
```bash
cd portal/frontend
npm install
npm run dev
```

## Configuration

Backend port: 8088
Frontend port: 3000 (dev), 5173 (Vite default)

Video paths are resolved relative to `training_data/`.

## Troubleshooting

### Portal shows no data
```bash
# Check JSON files exist
ls -lh analysis/*.json

# Test API
curl http://localhost:8088/api/clips | head
```

### Videos not playing
- Ensure videos are in `scenes/` or `vlm_copies/`
- Check file permissions
- Verify video path in API response

### CORS errors
Backend includes CORS middleware for localhost development.
