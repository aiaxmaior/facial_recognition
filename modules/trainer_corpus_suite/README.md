# Trainer Corpus Suite

A modular multimodal analysis pipeline for building high-quality training corpora. Designed to augment VLM captioning through sentiment analysis, emotion detection, demographic profiling, and action recognition.

## Vision

This suite is the foundation for a **dynamic ML stack** that:
- Orchestrates multiple analysis methods to extract rich semantic context from media
- Cross-validates outputs through clustering, keyword analysis, and multi-pass VLM confirmation
- Produces optimal reasoning outputs by combining sentiment, emotion, and action signals
- Generates training corpora suitable for both **diffusion LoRA fine-tuning** and **VLM instruction tuning**

## Core Capabilities

| Module | Purpose | Output |
|--------|---------|--------|
| **Person Detection** | Scene segmentation, pose estimation | Keypoints, bounding boxes |
| **Emotion Analysis** | Facial emotion + valence/arousal | Sentiment vectors |
| **Demographics** | Age/gender estimation, motion intensity | Profile metadata |
| **Action Recognition** | Human action classification (HAR) | Action labels + confidence |
| **VLM Captioning** | Natural language descriptions | Contextual captions |
| **Body Detection** | Region detection (configurable) | Detection labels |

## Pipeline Architecture

```
Media Input → Preprocessing → Scene Detection → Parallel Analysis → VLM Synthesis → Corpus Output
                                                      │
                    ┌─────────────┬─────────────┬─────┴─────┬─────────────┐
                    │             │             │           │             │
                 Emotion    Demographics    Action      Body        [Future]
                Detection    Analysis     Recognition  Detection    Modules
                    │             │             │           │             │
                    └─────────────┴─────────────┴─────┬─────┴─────────────┘
                                                      │
                                              Context Assembly
                                                      │
                                              VLM Captioning
                                                      │
                                            Training Corpus
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess media
python scripts/video_preprocessor.py

# 3. Run detection + scene splitting
python scripts/person_detector.py --i processed --scenes-dir scenes

# 4. Run analysis pipeline
./run_pipeline.sh

# 5. Launch review portal
cd portal && ./start.sh
```

## Configuration

Labels and detection categories are **configuration-driven** via `scripts/labels_config.json`:

```json
{
    "nudenet": {
        "enabled": true,
        "detection_labels": []
    },
    "har": {
        "enabled": true,
        "annotation_files": {...}
    }
}
```

This allows the pipeline to be adapted to different domains without code changes.

## Output Format

The final corpus (`unified_dataset.csv`) contains:

| Field | Description |
|-------|-------------|
| `media_path` | Path to processed clip |
| `caption` | VLM-generated description |
| `action` | Detected action label |
| `emotion` | Dominant emotion |
| `valence` | Emotional polarity (-1 to +1) |
| `arousal` | Activation level (0 to 1) |
| `quality_score` | Composite quality metric |

## Module Documentation

| Document | Description |
|----------|-------------|
| [docs/pipeline.md](docs/pipeline.md) | Full pipeline reference |
| [docs/modules.md](docs/modules.md) | Module API documentation |
| [docs/portal.md](docs/portal.md) | Web portal guide |

## Web Portal

Interactive data exploration and review interface:

```bash
cd portal && ./start.sh
# UI: http://localhost:3000
# API: http://localhost:8088/docs
```

Features:
- Video preview with annotation overlay
- Caption review and editing
- Filtering by emotion, action, demographics
- Batch export for training

## Future Direction

The suite is designed to evolve into a **self-optimizing ML stack**:

1. **Method Selection** - Automatically choose optimal analysis methods per input
2. **Cross-Validation** - Multiple VLM passes with consistency scoring
3. **Clustering** - Group similar outputs to identify patterns and outliers
4. **Reasoning Optimization** - Select best caption candidates through sentiment alignment
5. **Feedback Loop** - Learn from human corrections to improve future runs

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- FFmpeg for video processing
- Optional: Docker for HAR module

```bash
pip install -r requirements.txt
```

## License

MIT License - See LICENSE file.
