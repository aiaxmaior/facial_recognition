# Training Data Tools

Quick utility scripts for video processing and analysis.

## Available Tools

### `quick_nudenet_scan.py`
Quick nudity detection scanner for videos.

**Purpose**: Rapidly scan videos for nudity content without saving data.

**Usage**:
```bash
# Scan all videos in ./processed
python Tools/quick_nudenet_scan.py

# Scan specific videos
python Tools/quick_nudenet_scan.py --videos video1.mp4 video2.mp4

# Scan different directory
python Tools/quick_nudenet_scan.py --dir /path/to/videos
```

**Features**:
- Fast scanning with aggressive frame sampling (every 60th frame)
- Real-time feedback during scanning
- Summary report of nudity detection
- No data persistence - console output only

**Requirements**: NudeNet (`pip install nudenet`)

**Test Installation**: `python Tools/test_nudenet.py`

---

## Environment Setup

All tools expect to run in the `lora_trainer_suite` environment:

```bash
# Activate environment
conda activate lora_trainer_suite
# or: source Lora/lora_trainer_suite/venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Notes

- Tools are designed for quick analysis and debugging
- For production data processing, use scripts in the `scripts/` directory
- All tools work with videos in `./processed/` by default