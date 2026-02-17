# Module Reference

Detailed documentation for each processing module.

---

## video_preprocessor.py

**Purpose:** Standardizes heterogeneous video formats to consistent specification.

**Usage:**
```bash
python scripts/video_preprocessor.py [--input-dir DIR] [--output-dir DIR]
```

**Default Paths:**
- Input: `set_1/`
- Output: `processed/`

**Output Specification:**
| Parameter | Value |
|-----------|-------|
| Resolution | 960×544 |
| Codec | H.264 (libx264) |
| Quality | CRF 18 (visually lossless) |
| Frame Rate | 25 fps |
| Audio | Removed |
| Container | MP4 |

**Supported Input Formats:**
`.mp4`, `.MP4`, `.avi`, `.mkv`, `.flv`, `.m4v`, `.mov`, `.webm`

---

## person_detector.py

**Purpose:** Scene splitting, person detection, and pose extraction.

**Usage:**
```bash
python scripts/person_detector.py [--input-dir DIR] [--scenes-dir DIR] [--vlm-dir DIR]
```

**Default Paths:**
- Input: `processed/`
- Scenes: `scenes/`
- VLM copies: `vlm_copies/`
- Output: `analysis/detections.json`

**Detection Models:**
- Primary: YOLO26x (detection) + YOLO26x-pose (keypoints)
- Fallback: YOLOv8m + YOLOv8m-pose

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| FRAME_SAMPLE_RATE | 5 | Process every Nth frame |
| MIN_CONFIDENCE | 0.3 | Detection confidence threshold |
| MIN_COVERAGE | 0.60 | Min % frames with person |
| MIN_SCENE_LENGTH | 2s | Minimum scene duration |
| LOW_FPS_SHORT | 2 | FPS for scenes < 10s |
| LOW_FPS_LONG | 1 | FPS for scenes >= 10s |

**Pose Keypoints (17 per person):**
- 0: nose, 1-2: eyes, 3-4: ears
- 5-6: shoulders, 7-8: elbows, 9-10: wrists
- 11-12: hips, 13-14: knees, 15-16: ankles

**Outputs:**
- Scene clips in `scenes/` with naming: `{video}-Scene-{NNN}.mp4`
- Low-FPS copies in `vlm_copies/` for VLM processing
- Scene metadata CSV: `{video}_scenes.csv`
- Detection JSON with pose data

---

## emotion_detector.py

**Purpose:** Facial emotion recognition using DeepFace.

**Usage:**
```bash
python scripts/emotion_detector.py [--scenes-dir DIR] [--output FILE]
```

**Emotion Categories:**
| Emotion | Valence | Arousal |
|---------|---------|---------|
| Happy | +0.9 | 0.7 |
| Surprise | +0.3 | 0.9 |
| Neutral | 0.0 | 0.2 |
| Sad | -0.7 | 0.3 |
| Fear | -0.8 | 0.9 |
| Angry | -0.6 | 0.8 |
| Disgust | -0.7 | 0.5 |

**Output Metrics:**
- `dominant_emotion`: Most frequent emotion
- `emotion_distribution`: Percentage breakdown
- `mean_valence`: Average emotional polarity (-1 to +1)
- `mean_arousal`: Average activation level (0 to 1)
- `intensity_score`: |valence| × arousal

---

## demographics_detector.py

**Purpose:** Age/gender estimation and motion analysis.

**Usage:**
```bash
python scripts/demographics_detector.py [--scenes-dir DIR] [--output FILE]
```

**Age Categories:**
| Category | Age Range |
|----------|-----------|
| Infant | 0-3 |
| Child | 3-12 |
| Adolescent | 12-18 |
| Young Adult | 18-30 |
| Adult | 30-50 |
| Middle Aged | 50-65 |
| Senior | 65+ |

**Motion Analysis:**
- Uses Farneback optical flow
- Computes motion magnitude and direction variance
- Detects erratic movement (high magnitude + high directional variance)

**Output Metrics:**
- `mean_age`: Numeric age estimate
- `age_distribution`: Category counts
- `gender_distribution`: Male/Female counts
- `mean_motion`: Motion intensity
- `erratic_motion_ratio`: Proportion of erratic frames

---

## nudenet_batch_processor.py

**Purpose:** Content classification for filtering and labeling.

**Usage:**
```bash
python scripts/nudenet_batch_processor.py [--scenes-dir DIR] [--output FILE]
```

See [nudenet.md](nudenet.md) for detailed documentation.

---

## har_batch_analyzer.py

**Purpose:** Human Action Recognition using P-HAR models.

**Usage:**
```bash
python scripts/har_batch_analyzer.py [--scenes-dir DIR] [--max-clips N] [--device DEVICE]
```

**Note:** This script interfaces with P-HAR models. For full multimodal inference, use Docker. See [har.md](har.md) for details.

---

## vlm_captioner.py

**Purpose:** Generate natural language descriptions using VLM.

**Usage:**
```bash
python scripts/vlm_captioner.py [--input-dir DIR] [--output FILE]
```

**Requirements:**
- VLM server running at `localhost:8000`
- Processes `vlm_copies/` (low-FPS videos)

**Output:**
- Natural language caption per scene
- Structured metadata

---

## data_sanitizer.py

**Purpose:** Transform analysis data for AI-assisted review.

**Usage:**
```bash
python scripts/data_sanitizer.py [--input-dir DIR] [--output FILE]
```

**What it does:**
- Anonymizes/abstracts sensitive content descriptions
- Generates statistical summaries
- Allows AI assistants to help with data review without content restrictions

---

## final_assembler.py

**Purpose:** Combine all analysis outputs into unified dataset.

**Usage:**
```bash
python scripts/final_assembler.py [--analysis-dir DIR] [--output-dir DIR]
```

**Inputs:**
- `detections.json`
- `emotions.json`
- `demographics.json`
- `nudenet.json`
- `har_actions.json`
- `captions.json`

**Output:**
- `curated/unified_dataset.csv` - Training-ready dataset
- Quality scores and filtering applied

---

## Utility Scripts

### check_pipeline_status.py
Check current pipeline progress and status.

```bash
python scripts/check_pipeline_status.py
```

### robust_processor.py
Base class for incremental JSON writing with crash recovery.

Used by detection scripts for resumable processing.
