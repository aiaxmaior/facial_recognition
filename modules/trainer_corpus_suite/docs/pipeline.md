# Video Processing Pipeline

Complete pipeline for creating video diffusion model training data.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VIDEO DIFFUSION TRAINING PIPELINE                     │
└─────────────────────────────────────────────────────────────────────────────┘

  RAW VIDEOS          PREPROCESSING           SCENE ANALYSIS         CAPTIONING
  ──────────          ─────────────           ──────────────         ──────────
                                                                         
  set_1/        ───▶  processed/        ───▶  scenes/           ───▶  (video, caption)
  (mixed formats)     (960x544, H.264)        vlm_copies/             pairs
                                              detections.json         
                                                    │                      
                      ┌───────────────────────────────┼─────────────────────┐
                      ▼                       ▼       ▼         ▼           ▼
                  Emotion              Demographics  NudeNet   P-HAR      VLM
                  Detection            Detection     Analysis  (Docker)   Captioning
                      │                       │         │         │           │
                      ▼                       ▼         ▼         ▼           ▼
                  emotions.json      demographics.json nudenet.json har_actions.json captions.json
                                                    │
                                                    ▼
                                            Data Sanitizer
                                            (AI-safe review)
                                                    │
                                                    ▼
                                            Final Assembly
                                            unified_dataset.csv
```

## Pipeline Stages

### Stage 0: Video Preprocessing
**Script:** `python scripts/video_preprocessor.py`

Standardizes all input videos to consistent format:

| Parameter | Value |
|-----------|-------|
| Resolution | 960×544 |
| Codec | H.264 (libx264) |
| Quality | CRF 18 |
| Frame Rate | 25 fps |
| Audio | Stripped |
| Container | MP4 |

**Input:** `set_1/` (raw videos: .mp4, .avi, .mkv, .flv, etc.)
**Output:** `processed/`

---

### Stage 1: Person Detection + Scene Splitting
**Script:** `python scripts/person_detector.py --i processed --scenes-dir scenes --vlm-dir vlm_copies`

Performs three operations:
1. **Scene Splitting** - PySceneDetect splits videos into coherent scenes
2. **Person Detection** - YOLO26x detects humans in each scene
3. **Pose Extraction** - YOLO26x-pose extracts 17 body keypoints

**Detection Parameters:**
- Sample Rate: Every 5th frame
- Min Confidence: 0.3
- Min Coverage: 60% of frames must have person
- Min Scene Length: 2 seconds

**Outputs:**
- `scenes/` - Individual scene clips
- `vlm_copies/` - Low-FPS copies for VLM (2fps short, 1fps long)
- `analysis/detections.json` - Detection results with pose data

---

### Stage 2: Emotion Detection
**Script:** `python scripts/emotion_detector.py`

Analyzes facial expressions using DeepFace.

**Metrics:**
- Dominant emotion (happy, sad, angry, fear, surprise, neutral, disgust)
- Valence (-1 to +1): Emotional polarity
- Arousal (0 to 1): Activation level
- Intensity score: |valence| × arousal

**Output:** `analysis/emotions.json`

---

### Stage 3: Demographics Detection
**Script:** `python scripts/demographics_detector.py`

Extracts demographic attributes and motion patterns.

**Attributes:**
- Age estimation (categorical + numeric)
- Gender classification
- Motion intensity (optical flow)
- Erratic motion detection

**Output:** `analysis/demographics.json`

---

### Stage 4: NudeNet Analysis
**Script:** `python scripts/nudenet_batch_processor.py`

Content classification for filtering and labeling.

**Output:** `analysis/nudenet.json`

See [nudenet.md](nudenet.md) for details.

---

### Stage 5: P-HAR Action Recognition (Docker)
**Script:** `./run_har_analysis.sh` (runs in Docker)

Multimodal Human Action Recognition using P-HAR system.

**Models:**
- RGB (TimeSformer) - 17 action categories
- Skeleton (PoseC3D) - 6 categories
- Audio (ResNet101) - 4 categories

**Output:** `analysis/har_actions.json`

See [har.md](har.md) for details.

---

### Stage 6: VLM Captioning
**Script:** `python scripts/vlm_captioner.py`

Generates natural language descriptions using Vision-Language Model.

**Input:** `vlm_copies/` (low-fps scene videos)
**Output:** `analysis/captions.json`
**Requires:** VLM server at `localhost:8000`

---

### Stage 7: Data Sanitizer
**Script:** `python scripts/data_sanitizer.py`

Transforms data for AI-assisted review without content restrictions.

**Output:** `analysis/sanitized_stats.json`

---

### Stage 8: Final Assembly
**Script:** `python scripts/final_assembler.py`

Combines all analysis outputs into unified training dataset.

**Output:** `curated/unified_dataset.csv`

---

## Running the Pipeline

### Full Pipeline (Automated)
```bash
# Run main pipeline (emotion → demographics → nudenet → vlm → sanitizer → assembler)
./run_pipeline.sh

# Resume from specific step
./run_pipeline.sh --from 3

# Run specific step only
./run_pipeline.sh --only 2

# Skip VLM (if server not available)
./run_pipeline.sh --skip-vlm
```

### Manual Execution
```bash
# Stage 0: Preprocess videos
python scripts/video_preprocessor.py

# Stage 1: Detect persons and split scenes
python scripts/person_detector.py --i processed --scenes-dir scenes --vlm-dir vlm_copies

# Stage 2-4: Run pipeline
./run_pipeline.sh

# Stage 5: P-HAR (Docker - see har.md)
./run_har_analysis.sh

# Stage 6+: Continue pipeline
./run_pipeline.sh --from 4
```

---

## Directory Structure

```
training_data/
├── set_1/                    # Raw input videos
├── processed/                # Standardized videos
├── scenes/                   # Scene-split clips
├── vlm_copies/              # Low-FPS copies for VLM
├── analysis/                # All analysis outputs
│   ├── detections.json      # Person detection + pose
│   ├── emotions.json        # Facial emotion analysis
│   ├── demographics.json    # Age/gender/motion
│   ├── nudenet.json         # Content classification
│   ├── har_actions.json     # Action recognition (P-HAR)
│   ├── captions.json        # VLM descriptions
│   ├── sanitized_stats.json # AI-safe statistics
│   └── logs/                # Processing logs
├── curated/                 # Final dataset
│   └── unified_dataset.csv
├── scripts/                 # Processing scripts
├── portal/                  # Web visualization
├── docs/                    # Documentation
└── README.md
```

---

## Output Formats

### Training Data Format
Final `unified_dataset.csv` contains:

| Column | Description |
|--------|-------------|
| media_path | Path to scene video |
| caption | VLM-generated description |
| action | P-HAR action classification |
| emotion | Dominant emotion |
| valence | Emotional polarity |
| arousal | Activation level |
| age_estimate | Mean estimated age |
| gender | Dominant gender |
| quality_score | Composite quality metric |

This format is suitable for training text-to-video diffusion models.
