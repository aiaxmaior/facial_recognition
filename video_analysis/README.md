# Video Preprocessing & Emotion Analysis Pipeline

A comprehensive pipeline for preprocessing video datasets and performing automated emotion recognition analysis with validation.

---

## Purpose

This pipeline automates the process of:
- Standardizing heterogeneous video collections
- Detecting and isolating human presence in footage
- Analyzing emotional content through facial expression recognition
- Generating structured metadata for downstream applications
- Validating dataset quality and distribution

---

## Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raw Videos     │───▶│  Standardization│───▶│  Normalized     │
│  (mixed formats)│    │  Module         │    │  Video Files    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                       ┌──────────────────────────────┘
                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Human Presence │───▶│  Scene          │───▶│  Preview        │
│  Detection      │    │  Segments       │    │  Copies         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Emotion        │    │  Content        │
                       │  Recognition    │    │  Description    │
                       └─────────────────┘    └─────────────────┘
                              │                       │
                              └───────────┬───────────┘
                                          ▼
                              ┌─────────────────────┐
                              │  Analysis Module    │
                              │  - Distribution     │
                              │  - Clustering       │
                              │  - Validation       │
                              │  - Quality Scoring  │
                              └─────────────────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │  Validated Dataset  │
                              │  + Metadata         │
                              └─────────────────────┘
```

---

## Directory Structure

```
training_data/
├── set_1/                    # Source video files (various formats)
├── processed/                # Standardized video files
│   └── preprocessing_report.json
├── scenes/                   # Scene-segmented clips
├── vlm_copies/              # Low-framerate analysis copies
├── analysis/                # Analysis outputs
│   ├── detections.json      # Human detection results
│   ├── emotions.json        # Emotion recognition results
│   ├── captions.json        # Structured descriptions
│   ├── embeddings.npy       # Semantic embeddings
│   └── *.png                # Visualization outputs
├── curated/                 # Validated dataset
│   ├── dataset.json         # Final metadata
│   └── curation_report.md   # Quality report
├── scripts/                 # Processing scripts
├── notebooks/               # Analysis notebooks
└── requirements.txt         # Dependencies
```

---

## Processing Modules

### 1. Video Standardization

Normalizes all input videos to a consistent format for reliable downstream processing.

```bash
python scripts/video_preprocessor.py [--input-dir DIR] [--output-dir DIR]
```

**Specifications**:
| Parameter | Value |
|-----------|-------|
| Resolution | 960×544 |
| Codec | H.264 |
| Quality | CRF 18 |
| Frame Rate | 25 fps |
| Audio | Removed |

---

### 2. Human Presence Detection

Identifies scenes containing human subjects using object detection.

```bash
python scripts/person_detector.py [--input-dir DIR] [--scenes-dir DIR]
```

**Detection Parameters**:
- Model: YOLOv8 (medium)
- Target Class: Person
- Sampling Rate: Every 5th frame
- Presence Threshold: Detection in >60% of sampled frames
- Confidence Threshold: >0.3

**Outputs**:
- Scene-segmented video clips
- Low-framerate copies (1-2 fps) for detailed analysis
- Detection metadata (confidence, bounding box coverage)

---

### 3. Emotion Recognition

Analyzes facial expressions to derive emotional content metrics.

```bash
python scripts/emotion_detector.py [--scenes-dir DIR] [--output FILE]
```

**Recognition Framework**:

| Emotion Category | Valence | Arousal |
|------------------|---------|---------|
| Happy | +0.9 | 0.7 |
| Surprise | +0.3 | 0.9 |
| Neutral | 0.0 | 0.2 |
| Sad | -0.7 | 0.3 |
| Fear | -0.8 | 0.9 |
| Angry | -0.6 | 0.8 |
| Disgust | -0.7 | 0.5 |

**Derived Metrics**:

- **Valence**: Emotional polarity (-1 to +1)
  - Positive = Excitement, engagement, positive affect
  - Negative = Aversion, distress, negative affect

- **Arousal**: Activation level (0 to 1)
  - High = Intense, activated states
  - Low = Calm, subdued states

- **Intensity Score**: Combined metric (|valence| × arousal)
  - Captures emotional salience regardless of polarity

---

## Analysis Notebook

`notebooks/dataset_analysis.ipynb`

Interactive analysis environment with the following sections:

### Data Loading
- Import detection and emotion analysis results
- Merge datasets by scene identifier
- Filter to human-present segments

### Emotion Distribution Analysis
- Valence histogram (aversion ↔ excitement spectrum)
- Arousal histogram (calm ↔ activated spectrum)
- 2D valence-arousal scatter plot
- Dominant emotion frequency chart

### Semantic Analysis
- Structured description parsing
- Word frequency analysis by category
- Sentence embedding generation (all-MiniLM-L6-v2)

### Clustering
- HDBSCAN density-based clustering
- UMAP dimensionality reduction
- Cluster visualization and interpretation

### Quality Scoring
Composite score based on weighted criteria:

| Factor | Weight | Description |
|--------|--------|-------------|
| Emotional Intensity | 35% | High valence magnitude + high arousal |
| Subject Clarity | 30% | Detection confidence |
| Activity Level | 25% | Bounding box dynamics |
| Technical Quality | 10% | Detection coverage |

### Dataset Curation
- Priority-based selection
- Core/peripheral content balancing
- Export with structured metadata

---

## Emotion Model

### Valence-Arousal Space

```
                    High Arousal
                         │
        Fear/Anger ──────┼────── Excitement/Surprise
        (Aversion)       │       (Engagement)
                         │
   Low Valence ──────────┼──────────── High Valence
   (Negative)            │            (Positive)
                         │
        Sadness ─────────┼────── Contentment
        (Withdrawal)     │       (Calm Positive)
                         │
                    Low Arousal
```

### Intensity Calculation

```
intensity = |valence| × arousal
```

This captures emotionally salient content regardless of whether the affect is positive (excitement, engagement) or negative (aversion, distress).

---

## Output Formats

### Detection Results (`detections.json`)

```json
{
  "config": {
    "frame_sample_rate": 5,
    "min_confidence": 0.3,
    "min_coverage": 0.6
  },
  "summary": {
    "total_scenes": 150,
    "scenes_with_person": 120
  },
  "analyses": [
    {
      "scene_path": "scenes/video-Scene-001.mp4",
      "person_present": true,
      "detection_coverage": 0.85,
      "avg_confidence": 0.72,
      "max_persons_detected": 2
    }
  ]
}
```

### Emotion Results (`emotions.json`)

```json
{
  "analyses": [
    {
      "scene_path": "scenes/video-Scene-001.mp4",
      "dominant_emotion": "happy",
      "emotion_distribution": {
        "happy": 45.2,
        "neutral": 30.1,
        "surprise": 15.5
      },
      "mean_valence": 0.42,
      "mean_arousal": 0.65,
      "intensity_score": 0.27
    }
  ]
}
```

### Curated Dataset (`dataset.json`)

```json
[
  {
    "caption": "Structured scene description...",
    "media_path": "scenes/video-Scene-001.mp4",
    "metadata": {
      "valence": 0.42,
      "arousal": 0.65,
      "dominant_emotion": "happy",
      "priority_score": 0.78
    }
  }
]
```

---

## Quick Start

```bash
cd training_data

# Install dependencies
pip install -r requirements.txt

# Stage 1: Standardize videos
python scripts/video_preprocessor.py

# Stage 2: Detect humans and segment scenes
python scripts/person_detector.py

# Stage 3: Analyze emotions
python scripts/emotion_detector.py

# Stage 4: Interactive analysis
jupyter notebook notebooks/dataset_analysis.ipynb
```

---

## Dependencies

```
# Video Processing
av>=10.0.0
scenedetect[opencv]>=0.6.0

# Detection & Recognition
ultralytics>=8.0.0
deepface>=0.0.90

# NLP & Embeddings
sentence-transformers>=5.0.0

# Clustering
hdbscan>=0.8.29
umap-learn>=0.5.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.0.0

# Data Processing
pandas>=1.5.0
numpy>=1.21.0
tqdm>=4.64.0

# Notebook
jupyter>=1.0.0
```

---

## Validation Metrics

The pipeline generates the following validation outputs:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Detection Coverage | % of frames with human detected | >60% |
| Confidence Score | Mean detection confidence | >0.5 |
| Emotion Consistency | Variance in dominant emotion | Application-specific |
| Arousal Distribution | Spread across activation levels | Balanced preferred |
| Valence Distribution | Spread across polarity | Application-specific |

---

## Applications

This pipeline is suitable for:
- Emotion recognition dataset preparation
- Affective computing research
- Human behavior analysis
- Video content categorization
- Quality assurance for video collections
- Automated content filtering

---

## Notes

- All processing is fully automated without manual content review
- Analysis is based on model outputs and statistical aggregation
- The pipeline handles heterogeneous input formats
- Scene segmentation preserves temporal coherence
- Emotion metrics are derived from established psychological models
