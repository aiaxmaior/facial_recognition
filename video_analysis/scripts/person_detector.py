#!/usr/bin/env python3
"""
Person Detection and Scene Curation
====================================

Performs two key tasks:
1. Scene splitting using PySceneDetect
2. YOLOv8-based person detection to filter scenes with people

CRITICAL: This script processes files WITHOUT interpreting visual content.
All operations are based on model outputs and statistical metrics only.

Detection Strategy (face-agnostic, robust):
- Detects COCO class 0 (person) only
- Samples every Nth frame (N=5 for efficiency)
- Person is "present" if detected in >60% of sampled frames with confidence >0.3

Usage:
    python person_detector.py [--input-dir DIR] [--output-dir DIR]
"""

import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Detection parameters
FRAME_SAMPLE_RATE = 5  # Process every Nth frame
MIN_CONFIDENCE = 0.3   # Minimum detection confidence
MIN_COVERAGE = 0.60    # Minimum % of frames with person detected

# YOLO model path (use existing model in Lora suite)
YOLO_MODEL_PATH = Path(__file__).parent.parent.parent / "Lora/lora_trainer_suite/yolov8m.pt"

# Scene detection parameters
MIN_SCENE_LENGTH = "2s"  # Minimum scene length (50 frames at 25fps)

# Low-FPS copy parameters
LOW_FPS_SHORT = 2  # FPS for scenes < 10s
LOW_FPS_LONG = 1   # FPS for scenes >= 10s
LOW_FPS_THRESHOLD = 10  # Seconds threshold

# Default paths
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "processed"
DEFAULT_SCENES_DIR = Path(__file__).parent.parent / "scenes"
DEFAULT_VLM_DIR = Path(__file__).parent.parent / "vlm_copies"
DEFAULT_OUTPUT_FILE = Path(__file__).parent.parent / "analysis" / "detections.json"

# Video extensions to process
VIDEO_EXTENSIONS = {'.mp4', '.MP4'}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FrameDetection:
    """Detection results for a single frame."""
    frame_idx: int
    num_persons: int
    max_confidence: float
    total_bbox_area_ratio: float  # Sum of all person bbox areas / frame area


@dataclass
class SceneAnalysis:
    """Analysis results for a scene."""
    scene_path: str
    duration_seconds: float
    total_frames: int
    sampled_frames: int
    person_present: bool
    detection_coverage: float  # % of frames with person
    avg_confidence: float
    avg_bbox_area_ratio: float
    max_persons_detected: int
    frame_detections: List[Dict]  # Serialized FrameDetection list


# =============================================================================
# SCENE SPLITTING
# =============================================================================

def split_video_into_scenes(video_path: Path, output_dir: Path) -> List[Path]:
    """
    Split a video into scenes using PySceneDetect.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save scene clips
        
    Returns:
        List of paths to scene video files
    """
    # Use scenedetect CLI (more reliable than Python API for some edge cases)
    cmd = [
        'scenedetect',
        '-i', str(video_path),
        'detect-content',  # Content-based scene detection
        '-t', '27.0',  # Threshold (default is 27)
        'split-video',
        '-o', str(output_dir),
        '--filename', f'{video_path.stem}-Scene-$SCENE_NUMBER',
        'list-scenes',
        '-f', str(output_dir / f'{video_path.stem}_scenes.csv')
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout for long videos
        )
        
        # Find generated scene files
        scene_files = sorted(output_dir.glob(f"{video_path.stem}-Scene-*.mp4"))
        
        # Filter out scenes that are too short (if any slipped through)
        valid_scenes = []
        for scene_file in scene_files:
            # Quick duration check via ffprobe
            duration = get_video_duration(scene_file)
            if duration and duration >= 2.0:  # Minimum 2 seconds
                valid_scenes.append(scene_file)
            else:
                # Remove short scenes
                scene_file.unlink(missing_ok=True)
        
        return valid_scenes
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout splitting {video_path.name}")
        return []
    except Exception as e:
        print(f"  Error splitting {video_path.name}: {e}")
        return []


def get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except:
        return None


def get_video_frame_count(video_path: Path) -> int:
    """Get approximate frame count."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        frames = result.stdout.strip()
        if frames and frames != 'N/A':
            return int(frames)
    except:
        pass
    
    # Fallback: estimate from duration * fps
    duration = get_video_duration(video_path)
    if duration:
        return int(duration * 25)  # Assume 25 fps
    return 0


def generate_low_fps_copy(input_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Generate a low-FPS copy of a video for VLM analysis.
    
    Args:
        input_path: Path to source video
        output_dir: Directory to save low-FPS copy
        
    Returns:
        Path to generated file or None on failure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine FPS based on duration
    duration = get_video_duration(input_path)
    if duration is None:
        return None
    
    target_fps = LOW_FPS_LONG if duration >= LOW_FPS_THRESHOLD else LOW_FPS_SHORT
    
    output_path = output_dir / f"{input_path.stem}_vlm.mp4"
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', f'fps={target_fps}',
        '-c:v', 'libx264',
        '-crf', '23',  # Slightly lower quality OK for VLM
        '-preset', 'fast',
        '-an',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and output_path.exists():
            return output_path
    except:
        pass
    
    return None


# =============================================================================
# YOLO PERSON DETECTION
# =============================================================================

class PersonDetector:
    """YOLOv8-based person detector."""
    
    def __init__(self, model_path: Path = YOLO_MODEL_PATH):
        """Initialize YOLO model."""
        from ultralytics import YOLO
        
        self.model_path = model_path
        
        # Check if model exists, download default if not
        if not model_path.exists():
            print(f"Model not found at {model_path}, using default yolov8m.pt")
            self.model = YOLO('yolov8m.pt')
        else:
            self.model = YOLO(str(model_path))
        
        # COCO class 0 is "person"
        self.person_class = 0
    
    def detect_in_frame(self, frame: np.ndarray) -> FrameDetection:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            FrameDetection with results
        """
        # Run inference (only detect persons)
        results = self.model(frame, classes=[self.person_class], verbose=False)
        
        # Extract detections
        boxes = results[0].boxes if results else None
        
        if boxes is None or len(boxes) == 0:
            return FrameDetection(
                frame_idx=0,
                num_persons=0,
                max_confidence=0.0,
                total_bbox_area_ratio=0.0
            )
        
        # Calculate metrics
        frame_area = frame.shape[0] * frame.shape[1]
        total_bbox_area = 0
        max_conf = 0.0
        
        for box in boxes:
            conf = float(box.conf[0])
            if conf > MIN_CONFIDENCE:
                max_conf = max(max_conf, conf)
                # Calculate bbox area
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_area = (x2 - x1) * (y2 - y1)
                total_bbox_area += bbox_area
        
        return FrameDetection(
            frame_idx=0,
            num_persons=len([b for b in boxes if float(b.conf[0]) > MIN_CONFIDENCE]),
            max_confidence=max_conf,
            total_bbox_area_ratio=total_bbox_area / frame_area if frame_area > 0 else 0
        )
    
    def analyze_video(self, video_path: Path, sample_rate: int = FRAME_SAMPLE_RATE) -> SceneAnalysis:
        """
        Analyze a video for person presence.
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            
        Returns:
            SceneAnalysis with detection results
        """
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return SceneAnalysis(
                scene_path=str(video_path),
                duration_seconds=0,
                total_frames=0,
                sampled_frames=0,
                person_present=False,
                detection_coverage=0,
                avg_confidence=0,
                avg_bbox_area_ratio=0,
                max_persons_detected=0,
                frame_detections=[]
            )
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total_frames / fps
        
        # Process frames
        detections = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every Nth frame
            if frame_idx % sample_rate == 0:
                detection = self.detect_in_frame(frame)
                detection.frame_idx = frame_idx
                detections.append(detection)
            
            frame_idx += 1
        
        cap.release()
        
        # Aggregate results
        if not detections:
            return SceneAnalysis(
                scene_path=str(video_path),
                duration_seconds=duration,
                total_frames=total_frames,
                sampled_frames=0,
                person_present=False,
                detection_coverage=0,
                avg_confidence=0,
                avg_bbox_area_ratio=0,
                max_persons_detected=0,
                frame_detections=[]
            )
        
        # Calculate metrics
        frames_with_person = [d for d in detections if d.num_persons > 0]
        detection_coverage = len(frames_with_person) / len(detections)
        
        avg_confidence = (
            np.mean([d.max_confidence for d in frames_with_person])
            if frames_with_person else 0
        )
        
        avg_bbox_area = (
            np.mean([d.total_bbox_area_ratio for d in frames_with_person])
            if frames_with_person else 0
        )
        
        max_persons = max(d.num_persons for d in detections)
        
        # Determine if person is present based on coverage threshold
        person_present = detection_coverage >= MIN_COVERAGE
        
        return SceneAnalysis(
            scene_path=str(video_path),
            duration_seconds=round(duration, 2),
            total_frames=total_frames,
            sampled_frames=len(detections),
            person_present=person_present,
            detection_coverage=round(detection_coverage, 3),
            avg_confidence=round(avg_confidence, 3),
            avg_bbox_area_ratio=round(avg_bbox_area, 4),
            max_persons_detected=max_persons,
            frame_detections=[asdict(d) for d in detections]
        )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def discover_videos(input_dir: Path) -> List[Path]:
    """Find all video files in input directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_dir.glob(f"*{ext}"))
    return sorted(videos)


def process_all(input_dir: Path, scenes_dir: Path, vlm_dir: Path, output_file: Path) -> Dict:
    """
    Process all videos: split into scenes, detect persons, generate VLM copies.
    
    Args:
        input_dir: Directory with preprocessed videos
        scenes_dir: Directory to save scene clips
        vlm_dir: Directory to save low-FPS VLM copies
        output_file: Path to save detection results JSON
        
    Returns:
        Summary statistics
    """
    # Ensure directories exist
    scenes_dir.mkdir(parents=True, exist_ok=True)
    vlm_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Discover videos
    videos = discover_videos(input_dir)
    print(f"Found {len(videos)} preprocessed videos")
    
    if not videos:
        print("No videos found to process")
        return {'total_videos': 0, 'total_scenes': 0, 'scenes_with_person': 0}
    
    # Initialize detector
    print("Loading YOLO model...")
    detector = PersonDetector()
    
    # Track results
    all_analyses = []
    total_scenes = 0
    scenes_with_person = 0
    vlm_copies_created = 0
    
    # Process each video
    for video_path in tqdm(videos, desc="Processing videos"):
        print(f"\n  Processing: {video_path.name}")
        
        # Step 1: Split into scenes
        print(f"    Splitting into scenes...")
        scene_files = split_video_into_scenes(video_path, scenes_dir)
        print(f"    Found {len(scene_files)} scenes")
        
        if not scene_files:
            # If scene splitting fails, treat whole video as one scene
            scene_files = [video_path]
        
        # Step 2: Analyze each scene
        for scene_path in tqdm(scene_files, desc="    Analyzing scenes", leave=False):
            analysis = detector.analyze_video(scene_path)
            all_analyses.append(asdict(analysis))
            total_scenes += 1
            
            if analysis.person_present:
                scenes_with_person += 1
                
                # Step 3: Generate low-FPS copy for VLM
                vlm_path = generate_low_fps_copy(scene_path, vlm_dir)
                if vlm_path:
                    vlm_copies_created += 1
    
    # Save results
    results = {
        'config': {
            'frame_sample_rate': FRAME_SAMPLE_RATE,
            'min_confidence': MIN_CONFIDENCE,
            'min_coverage': MIN_COVERAGE,
            'model_path': str(YOLO_MODEL_PATH),
            'low_fps_short': LOW_FPS_SHORT,
            'low_fps_long': LOW_FPS_LONG,
            'low_fps_threshold': LOW_FPS_THRESHOLD
        },
        'summary': {
            'total_videos': len(videos),
            'total_scenes': total_scenes,
            'scenes_with_person': scenes_with_person,
            'scenes_without_person': total_scenes - scenes_with_person,
            'vlm_copies_created': vlm_copies_created
        },
        'analyses': all_analyses
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results['summary']


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split videos into scenes and detect persons",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with preprocessed videos (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        '--scenes-dir', '-s',
        type=Path,
        default=DEFAULT_SCENES_DIR,
        help=f"Directory to save scene clips (default: {DEFAULT_SCENES_DIR})"
    )
    
    parser.add_argument(
        '--vlm-dir', '-v',
        type=Path,
        default=DEFAULT_VLM_DIR,
        help=f"Directory to save low-FPS VLM copies (default: {DEFAULT_VLM_DIR})"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})"
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Print config
    print("=" * 60)
    print("Person Detection and Scene Curation")
    print("=" * 60)
    print(f"Input:     {args.input_dir}")
    print(f"Scenes:    {args.scenes_dir}")
    print(f"VLM Dir:   {args.vlm_dir}")
    print(f"Output:    {args.output}")
    print(f"Sample Rate: Every {FRAME_SAMPLE_RATE} frames")
    print(f"Min Confidence: {MIN_CONFIDENCE}")
    print(f"Min Coverage: {MIN_COVERAGE*100:.0f}%")
    print(f"VLM FPS: {LOW_FPS_SHORT} (short) / {LOW_FPS_LONG} (long)")
    print("=" * 60)
    
    # Process
    summary = process_all(args.input_dir, args.scenes_dir, args.vlm_dir, args.output)
    
    # Report
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Videos:   {summary['total_videos']}")
    print(f"Total Scenes:   {summary['total_scenes']}")
    print(f"With Person:    {summary['scenes_with_person']}")
    print(f"Without Person: {summary['scenes_without_person']}")
    print(f"VLM Copies:     {summary.get('vlm_copies_created', 0)}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
