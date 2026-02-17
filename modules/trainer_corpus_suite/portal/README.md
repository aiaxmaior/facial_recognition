# Pipeline Data Portal

A sleek, modern dark grey + cerulean blue React/TypeScript portal for exploring video analysis pipeline output.

## Features

- **Insights Dashboard** - Key metrics, emotion distribution charts, valence/arousal scatter plots
- **Clips Table** - Interactive table with sorting, filtering by video/emotion/person detection
- **Videos Table** - Aggregated video-level statistics with emotion breakdowns
- **Detail Modals** - Click any row to see full data including emotion distribution, demographics, captions
- **Video Preview** - Built-in video player modal for previewing scene clips and source videos

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite
- **UI Framework**: Mantine 7, Mantine DataTable
- **Charts**: Recharts
- **Icons**: Tabler Icons
- **Backend**: FastAPI, Python 3.11+

## Quick Start

```bash
# Make start script executable
chmod +x start.sh

# Start both backend and frontend
./start.sh
```

Or start them separately:

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python main.py

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

## URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8088 |
| API Docs | http://localhost:8088/docs |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/pipeline-data` | All data (clips, videos, insights) |
| `GET /api/clips` | All clip data |
| `GET /api/videos` | Video summaries |
| `GET /api/insights` | Pipeline insights/stats |
| `GET /api/clip/{scene_name}` | Single clip details |
| `GET /api/video/{video_name}` | Single video summary |
| `GET /api/video/scene/{scene_name}` | Stream scene clip video |
| `GET /api/video/source/{video_name}` | Stream source video |
| `GET /api/video/vlm/{scene_name}` | Stream VLM copy video |
| `GET /api/video/exists/{type}/{name}` | Check if video exists |
| `GET /api/raw/detections` | Raw detections.json |
| `GET /api/raw/emotions` | Raw emotions.json |
| `GET /api/raw/captions` | Raw captions.json |

## Data Sources

The portal reads from these JSON files in `training_data/analysis/`:

- `detections.json` - Person detection results
- `emotions.json` - Emotion analysis results
- `captions.json` - VLM caption results

And from `training_data/processed/data/`:

- `processing_summary.json` - Processing statistics

## Theme

- **Primary**: Cerulean Blue (`#3395f3`)
- **Background**: Dark Grey gradient (`#0d1117` → `#161b22`)
- **Accent Glow**: Cerulean with subtle bloom effects
- **Typography**: Outfit (headings), JetBrains Mono (code)

## Screenshots

Coming soon...
