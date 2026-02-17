#!/usr/bin/env python3
"""
NudeNet Batch Video Processor
=============================

Processes all videos in ./processed through NudeNet detection and saves results
to parquet files using PyArrow for efficient data storage.

Key Features:
- Runs NudeNet detection on sampled video frames
- Efficient data collection with PyArrow during processing loop
- Saves results to ./processed/data as parquet files
- Designed for conda environment 'lora_trainer_suite'

Usage:
    conda activate lora_trainer_suite
    python nudenet_batch_processor.py

Output:
- Creates ./processed/data directory with parquet files
- Each video gets a separate parquet file with detection results
- Consolidated summary parquet with all results
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

# PyArrow for efficient data handling
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyarrow import Table
except ImportError:
    print("ERROR: PyArrow not found. Please install with: pip install pyarrow")
    print("Or run in lora_trainer_suite environment: source Lora/lora_trainer_suite/venv/bin/activate")
    sys.exit(1)

# NudeNet for detection
try:
    import nudenet as nn
except ImportError:
    print("ERROR: NudeNet not found. Please install with: pip install nudenet")
    print("Or run in lora_trainer_suite environment: source Lora/lora_trainer_suite/venv/bin/activate")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output directories (defaults, can be overridden via CLI)
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "scenes"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "analysis" / "nudenet_data"
DEFAULT_JSON_OUTPUT = Path(__file__).parent.parent / "analysis" / "nudenet.json"

# Legacy paths for backward compatibility
PROCESSED_DIR = Path(__file__).parent.parent / "processed"
OUTPUT_DATA_DIR = PROCESSED_DIR / "data"

# Detection parameters
FRAME_SAMPLE_RATE = 30  # Process every Nth frame (for efficiency)
NUDENET_MODEL_SIZE = 640  # 640px model (balanced speed/accuracy)

# Detection labels loaded from configuration
# Labels are domain-specific and loaded at runtime from labels_config.json
# This keeps the codebase clean and allows easy domain customization
LABELS_CONFIG_FILE = Path(__file__).parent / "labels_config.json"

def _load_detection_labels() -> list:
    """Load detection labels from configuration file."""
    if LABELS_CONFIG_FILE.exists():
        with open(LABELS_CONFIG_FILE) as f:
            config = json.load(f)
            return config.get('nudenet', {}).get('detection_labels', [])
    return []

# Labels loaded at module import - empty list means detect all available labels
DETECTION_LABELS = _load_detection_labels()

# Video extensions to process
VIDEO_EXTENSIONS = {'.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV'}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class FrameDetection:
    """NudeNet detection results for a single frame."""

    def __init__(self, video_path: str, frame_idx: int, timestamp: float,
                 detections: List[Dict[str, Any]]):
        self.video_path = video_path
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.detections = detections
        self.num_detections = len(detections)

        # Extract key metrics
        self.labels_found = [d.get('label', '') for d in detections]
        self.confidences = [d.get('score', 0.0) for d in detections]
        self.max_confidence = max(self.confidences) if self.confidences else 0.0
        self.avg_confidence = np.mean(self.confidences) if self.confidences else 0.0

        # Check for exposed categories
        exposed_labels = [l for l in self.labels_found if 'EXPOSED' in l]
        self.has_exposed_content = len(exposed_labels) > 0
        self.exposed_labels = exposed_labels

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PyArrow."""
        return {
            'video_path': self.video_path,
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'num_detections': self.num_detections,
            'labels_found': self.labels_found,
            'confidences': self.confidences,
            'max_confidence': self.max_confidence,
            'avg_confidence': self.avg_confidence,
            'has_exposed_content': self.has_exposed_content,
            'exposed_labels': self.exposed_labels,
            'detections_json': json.dumps(self.detections),  # Store full detections as JSON
        }


class VideoAnalysis:
    """Analysis results for an entire video."""

    def __init__(self, video_path: str, total_frames: int, sampled_frames: int,
                 total_detections: int, frames_with_detections: int,
                 frames_with_exposed: int, processing_time: float):
        self.video_path = str(video_path)
        self.video_name = video_path.name
        self.total_frames = total_frames
        self.sampled_frames = sampled_frames
        self.total_detections = total_detections
        self.frames_with_detections = frames_with_detections
        self.frames_with_exposed = frames_with_exposed
        self.detection_rate = frames_with_detections / sampled_frames if sampled_frames > 0 else 0
        self.exposed_rate = frames_with_exposed / sampled_frames if sampled_frames > 0 else 0
        self.processing_time = processing_time
        self.processed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PyArrow."""
        return {
            'video_path': self.video_path,
            'video_name': self.video_name,
            'total_frames': self.total_frames,
            'sampled_frames': self.sampled_frames,
            'total_detections': self.total_detections,
            'frames_with_detections': self.frames_with_detections,
            'frames_with_exposed': self.frames_with_exposed,
            'detection_rate': self.detection_rate,
            'exposed_rate': self.exposed_rate,
            'processing_time': self.processing_time,
            'processed_at': self.processed_at,
        }


# =============================================================================
# NUDENET DETECTOR
# =============================================================================

class NudeNetDetector:
    """NudeNet-based detector for batch video processing."""

    def __init__(self, model_size: int = NUDENET_MODEL_SIZE):
        """Initialize NudeNet model."""
        print(f"Loading NudeNet model (size: {model_size}px)...")
        from nudenet import NudeDetector
        self.model = NudeDetector()
        print("✓ NudeNet model loaded")

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run NudeNet detection on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detections with class and score
        """
        # Convert BGR to RGB for NudeNet
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        detections = self.model.detect(rgb_frame)

        # Filter to our labels of interest
        filtered_detections = []
        for detection in detections:
            # NudeNet returns 'class' not 'label'
            label = detection.get('class', '')
            if label in DETECTION_LABELS:
                filtered_detections.append({
                    'label': label,
                    'score': detection.get('score', 0.0),
                    'box': detection.get('box', []),
                })

        return filtered_detections

    def analyze_video(self, video_path: Path, sample_rate: int = FRAME_SAMPLE_RATE) -> tuple:
        """
        Analyze a video for nudity content.

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame

        Returns:
            Tuple of (list of FrameDetection objects, VideoAnalysis object)
        """
        import time
        start_time = time.time()

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"  Warning: Could not open video {video_path.name}")
            return [], None

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        frame_detections = []
        frame_idx = 0
        processed_count = 0

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps

                # Run detection
                detections = self.detect_frame(frame)

                # Create FrameDetection object
                frame_detection = FrameDetection(
                    video_path=str(video_path),
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    detections=detections
                )

                frame_detections.append(frame_detection)
                processed_count += 1

            frame_idx += 1

        cap.release()

        # Calculate video-level statistics
        total_detections = sum(fd.num_detections for fd in frame_detections)
        frames_with_detections = sum(1 for fd in frame_detections if fd.num_detections > 0)
        frames_with_exposed = sum(1 for fd in frame_detections if fd.has_exposed_content)

        processing_time = time.time() - start_time

        video_analysis = VideoAnalysis(
            video_path=video_path,
            total_frames=total_frames,
            sampled_frames=len(frame_detections),
            total_detections=total_detections,
            frames_with_detections=frames_with_detections,
            frames_with_exposed=frames_with_exposed,
            processing_time=processing_time
        )

        return frame_detections, video_analysis


# =============================================================================
# DATA EXPORT UTILITIES
# =============================================================================

def save_frame_detections_to_parquet(frame_detections: List[FrameDetection],
                                   output_path: Path) -> None:
    """
    Save frame-level detections to parquet using PyArrow.

    Args:
        frame_detections: List of FrameDetection objects
        output_path: Path to save parquet file
    """
    if not frame_detections:
        return

    # Convert to list of dictionaries
    data_dicts = [fd.to_dict() for fd in frame_detections]

    # Create PyArrow table
    df = pd.DataFrame(data_dicts)

    # Convert lists to strings for parquet compatibility
    df['labels_found'] = df['labels_found'].apply(lambda x: ','.join(x))
    df['confidences'] = df['confidences'].apply(lambda x: ','.join(map(str, x)))
    df['exposed_labels'] = df['exposed_labels'].apply(lambda x: ','.join(x))

    # Convert to PyArrow table
    table = pa.Table.from_pandas(df)

    # Write to parquet
    pq.write_table(table, output_path, compression='snappy')


def save_video_summaries_to_parquet(video_analyses: List[VideoAnalysis],
                                  output_path: Path) -> None:
    """
    Save video-level summaries to parquet.

    Args:
        video_analyses: List of VideoAnalysis objects
        output_path: Path to save parquet file
    """
    if not video_analyses:
        return

    # Convert to list of dictionaries
    data_dicts = [va.to_dict() for va in video_analyses]

    # Create PyArrow table
    df = pd.DataFrame(data_dicts)
    table = pa.Table.from_pandas(df)

    # Write to parquet
    pq.write_table(table, output_path, compression='snappy')


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def discover_videos(input_dir: Path) -> List[Path]:
    """Find all video files in input directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_dir.glob(f"*{ext}"))
    return sorted(videos)


def process_all_videos(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process all videos in input directory through NudeNet.

    Args:
        input_dir: Directory with videos to process
        output_dir: Directory to save parquet results

    Returns:
        Summary statistics
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos
    videos = discover_videos(input_dir)
    print(f"Found {len(videos)} videos to process")

    if not videos:
        print("No videos found to process")
        return {'total_videos': 0}

    # Initialize detector
    detector = NudeNetDetector()

    # Track results
    all_frame_detections = []
    all_video_analyses = []

    # Process each video
    for video_path in tqdm(videos, desc="Processing videos"):
        print(f"\n  Processing: {video_path.name}")

        # Analyze video
        frame_detections, video_analysis = detector.analyze_video(video_path)

        if video_analysis:
            all_video_analyses.append(video_analysis)

            # Save individual video results
            video_name = video_path.stem
            frame_parquet_path = output_dir / f"{video_name}_frames.parquet"
            save_frame_detections_to_parquet(frame_detections, frame_parquet_path)

            print(f"    Frames processed: {video_analysis.sampled_frames}")
            print(f"    Total detections: {video_analysis.total_detections}")
            print(f"    Frames with exposed content: {video_analysis.frames_with_exposed}")
            print(f"    Results saved to: {frame_parquet_path}")

            # Accumulate frame detections for consolidated file
            all_frame_detections.extend(frame_detections)

    # Save consolidated results
    print(f"\nSaving consolidated results...")

    # Save all frame detections
    all_frames_path = output_dir / "all_frames.parquet"
    save_frame_detections_to_parquet(all_frame_detections, all_frames_path)

    # Save video summaries
    video_summaries_path = output_dir / "video_summaries.parquet"
    save_video_summaries_to_parquet(all_video_analyses, video_summaries_path)

    # Create summary statistics
    summary = {
        'total_videos': len(videos),
        'total_frames_processed': sum(va.sampled_frames for va in all_video_analyses),
        'total_detections': sum(va.total_detections for va in all_video_analyses),
        'videos_with_detections': sum(1 for va in all_video_analyses if va.frames_with_detections > 0),
        'videos_with_exposed': sum(1 for va in all_video_analyses if va.frames_with_exposed > 0),
        'avg_detection_rate': np.mean([va.detection_rate for va in all_video_analyses]) if all_video_analyses else 0,
        'avg_exposed_rate': np.mean([va.exposed_rate for va in all_video_analyses]) if all_video_analyses else 0,
        'total_processing_time': sum(va.processing_time for va in all_video_analyses),
        'output_directory': str(output_dir),
        'processed_at': datetime.now().isoformat(),
    }

    # Save summary as JSON
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def export_to_json(output_dir: Path, json_output: Path) -> None:
    """
    Export parquet results to JSON format for VLM context integration.
    
    Args:
        output_dir: Directory containing parquet files
        json_output: Path for JSON output file
    """
    all_frames_parquet = output_dir / "all_frames.parquet"
    
    if not all_frames_parquet.exists():
        print(f"Warning: {all_frames_parquet} not found, skipping JSON export")
        return
        
    # Load parquet data
    frames_df = pd.read_parquet(all_frames_parquet)
    
    # Group by video/scene and aggregate
    scene_data = {}
    for video_path, group in frames_df.groupby('video_path'):
        scene_name = Path(video_path).stem
        
        # Aggregate detections per scene
        all_labels = []
        for _, row in group.iterrows():
            if pd.notna(row.get('detected_labels')) and row['detected_labels']:
                labels = row['detected_labels'].split(',') if isinstance(row['detected_labels'], str) else []
                all_labels.extend(labels)
        
        # Count label frequencies
        label_counts = {}
        for label in all_labels:
            label = label.strip()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        scene_data[scene_name] = {
            'scene_path': str(video_path),
            'total_frames_sampled': len(group),
            'frames_with_detections': int(group['has_detection'].sum()) if 'has_detection' in group.columns else 0,
            'label_counts': label_counts,
            'detection_rate': float(group['has_detection'].mean()) if 'has_detection' in group.columns else 0.0
        }
    
    # Write JSON
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, 'w') as f:
        json.dump({
            'total_scenes': len(scene_data),
            'analyses': scene_data
        }, f, indent=2)
    
    print(f"JSON export saved to: {json_output}")


def main():
    """Main entry point with CLI argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NudeNet Batch Video Processor - Detect body parts in video frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process scenes directory (default)
  python nudenet_batch_processor.py
  
  # Process specific directory
  python nudenet_batch_processor.py --input-dir processed/ --output-dir analysis/nudenet_processed
  
  # Process and export JSON for VLM context
  python nudenet_batch_processor.py --json-output analysis/nudenet.json
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with video files (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for parquet output (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        '--json-output', '-j',
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"JSON output file for VLM context (default: {DEFAULT_JSON_OUTPUT})"
    )
    
    parser.add_argument(
        '--sample-rate', '-r',
        type=int,
        default=FRAME_SAMPLE_RATE,
        help=f"Process every Nth frame (default: {FRAME_SAMPLE_RATE})"
    )
    
    parser.add_argument(
        '--skip-json', action='store_true',
        help="Skip JSON export (only output parquet)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NudeNet Batch Video Processor")
    print("=" * 60)
    print(f"Input Directory:  {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"JSON Output:      {args.json_output}")
    print(f"Frame Sample Rate: Every {args.sample_rate} frames")
    print(f"NudeNet Model Size: {NUDENET_MODEL_SIZE}px")
    print(f"Labels: {len(DETECTION_LABELS)} categories")
    print("=" * 60)

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Check environment
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - processing will be slower")
    except ImportError:
        print("⚠ PyTorch not found - some features may not work")

    # Process videos
    summary = process_all_videos(args.input_dir, args.output_dir)

    # Report results
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total Videos:     {summary['total_videos']}")
    print(f"Frames Processed: {summary['total_frames_processed']}")
    print(f"Total Detections: {summary['total_detections']}")
    print(f"Videos w/ Content: {summary['videos_with_detections']}")
    print(f"Videos w/ Exposed: {summary['videos_with_exposed']}")
    print(f"Output Location:  {summary['output_directory']}")
    print("=" * 60)

    print("\nOutput files:")
    print(f"  - all_frames.parquet: All frame-level detections")
    print(f"  - video_summaries.parquet: Video-level summaries")
    print(f"  - processing_summary.json: Processing statistics")
    print(f"  - [video_name]_frames.parquet: Individual video results")
    
    # Export to JSON for VLM context integration
    if not args.skip_json:
        print("\nExporting to JSON for VLM context...")
        export_to_json(args.output_dir, args.json_output)


if __name__ == "__main__":
    main()