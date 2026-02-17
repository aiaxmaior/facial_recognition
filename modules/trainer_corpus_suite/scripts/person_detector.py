#!/usr/bin/env python3
"""
Person Detection and Scene Curation
====================================

Performs three key tasks:
1. Scene splitting using PySceneDetect
2. YOLO26x-based person detection to filter scenes with people
3. YOLO26x-pose for body keypoint extraction and interaction analysis

CRITICAL: This script processes files WITHOUT interpreting visual content.
All operations are based on model outputs and statistical metrics only.

Detection Strategy (face-agnostic, robust):
- Detects COCO class 0 (person) only
- Extracts 17 body keypoints per person (pose estimation)
- Samples every Nth frame (N=5 for efficiency)
- Person is "present" if detected in >60% of sampled frames with confidence >0.3

Pose Keypoints (17 per person):
- 0: nose, 1-2: eyes, 3-4: ears
- 5-6: shoulders, 7-8: elbows, 9-10: wrists
- 11-12: hips, 13-14: knees, 15-16: ankles

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
import torch

# =============================================================================
# PyTorch 2.6 COMPATIBILITY FIX
# =============================================================================
# PyTorch 2.6 changed torch.load to use weights_only=True by default
# This breaks loading YOLO/ultralytics weights. We need to allowlist the classes.
try:
    from ultralytics.nn.tasks import DetectionModel, PoseModel, SegmentationModel
    torch.serialization.add_safe_globals([DetectionModel, PoseModel, SegmentationModel])
except ImportError:
    pass  # Will handle in model loading

# Alternative: Monkey-patch torch.load for ultralytics compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for .pt files (YOLO weights)
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply patch
torch.load = _patched_torch_load

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Detection parameters
FRAME_SAMPLE_RATE = 5  # Process every Nth frame
MIN_CONFIDENCE = 0.3   # Minimum detection confidence
MIN_COVERAGE = 0.60    # Minimum % of frames with person detected

# Weights directory (centralized for all models)
WEIGHTS_DIR = Path(__file__).parent / "weights"

# YOLO model paths (YOLO26x for detection, YOLO26x-pose for keypoints)
YOLO_DETECT_MODEL = WEIGHTS_DIR / "yolo26x.pt"      # Detection model
YOLO_POSE_MODEL = WEIGHTS_DIR / "yolo26x-pose.pt"   # Pose estimation model
YOLO_MODEL_PATH = WEIGHTS_DIR / "yolov8m.pt"        # Legacy fallback

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
class PersonKeypoints:
    """Body keypoints for a single detected person."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    # 17 keypoints: [x, y, confidence] for each
    keypoints: List[List[float]]  # Shape: (17, 3)
    # Derived pose metrics
    body_orientation: str  # 'front', 'back', 'left', 'right'
    posture: str  # 'standing', 'sitting', 'lying', 'unknown'
    arm_position: str  # 'down', 'raised', 'extended', 'unknown'


@dataclass
class FrameDetection:
    """Detection results for a single frame."""
    frame_idx: int
    num_persons: int
    max_confidence: float
    total_bbox_area_ratio: float  # Sum of all person bbox areas / frame area
    # Pose data (new)
    persons: List[Dict] = None  # List of PersonKeypoints as dicts
    interpersonal_distance: float = -1.0  # Min distance between people (-1 if <2 people)
    interaction_detected: bool = False  # True if people facing each other / close proximity


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
    # Pose analysis summary (new)
    pose_summary: Dict = None  # Aggregated pose statistics for the scene


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
    # Use --copy for stream copy (preserves audio if present)
    cmd = [
        'scenedetect',
        '-i', str(video_path),
        'detect-content',  # Content-based scene detection
        '-t', '27.0',  # Threshold (default is 27)
        'split-video',
        '-o', str(output_dir),
        '--copy',  # Stream copy - preserves audio
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
# POSE ANALYSIS UTILITIES
# =============================================================================

# Keypoint indices
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def analyze_body_orientation(keypoints: np.ndarray) -> str:
    """
    Determine body orientation from keypoints.
    
    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf] per keypoint
        
    Returns:
        'front', 'back', 'left', 'right', or 'unknown'
    """
    # Get shoulder positions (indices 5, 6)
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    nose = keypoints[0]
    
    # Check confidence
    if left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
        return 'unknown'
    
    # Calculate shoulder width and midpoint
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    
    # If shoulders are very close horizontally, person is facing sideways
    if shoulder_width < 30:  # Narrow shoulder width = profile view
        if nose[2] > 0.3:
            # Nose visible and to one side = facing that direction
            if nose[0] < shoulder_mid_x:
                return 'left'
            else:
                return 'right'
        return 'left' if left_shoulder[0] < right_shoulder[0] else 'right'
    
    # Check if nose is visible (front) or not (back)
    if nose[2] > 0.5:
        return 'front'
    elif nose[2] < 0.2:
        return 'back'
    
    return 'front'  # Default assumption


def analyze_posture(keypoints: np.ndarray) -> str:
    """
    Determine body posture from keypoints.
    
    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf] per keypoint
        
    Returns:
        'standing', 'sitting', 'lying', or 'unknown'
    """
    # Get key points
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    # Check confidence on critical points
    hip_conf = min(left_hip[2], right_hip[2])
    knee_conf = min(left_knee[2], right_knee[2])
    shoulder_conf = min(left_shoulder[2], right_shoulder[2])
    
    if hip_conf < 0.3 or shoulder_conf < 0.3:
        return 'unknown'
    
    # Calculate vertical distances
    hip_y = (left_hip[1] + right_hip[1]) / 2
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    torso_height = abs(hip_y - shoulder_y)
    
    if knee_conf > 0.3:
        knee_y = (left_knee[1] + right_knee[1]) / 2
        hip_knee_dist = abs(knee_y - hip_y)
        
        # Lying: torso is more horizontal than vertical
        if torso_height < 50 and abs(left_shoulder[1] - left_hip[1]) < 100:
            return 'lying'
        
        # Sitting: knees are at similar height to hips
        if hip_knee_dist < torso_height * 0.5:
            return 'sitting'
    
    # Default to standing
    return 'standing'


def analyze_arm_position(keypoints: np.ndarray) -> str:
    """
    Determine arm position from keypoints.
    
    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf] per keypoint
        
    Returns:
        'down', 'raised', 'extended', or 'unknown'
    """
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    # Check confidence
    if left_shoulder[2] < 0.3 and right_shoulder[2] < 0.3:
        return 'unknown'
    
    arms_raised = False
    arms_extended = False
    
    # Check left arm
    if left_wrist[2] > 0.3 and left_shoulder[2] > 0.3:
        if left_wrist[1] < left_shoulder[1]:  # Wrist above shoulder
            arms_raised = True
        if abs(left_wrist[0] - left_shoulder[0]) > 100:  # Wrist far from shoulder horizontally
            arms_extended = True
    
    # Check right arm
    if right_wrist[2] > 0.3 and right_shoulder[2] > 0.3:
        if right_wrist[1] < right_shoulder[1]:
            arms_raised = True
        if abs(right_wrist[0] - right_shoulder[0]) > 100:
            arms_extended = True
    
    if arms_raised:
        return 'raised'
    elif arms_extended:
        return 'extended'
    
    return 'down'


def calculate_interpersonal_distance(persons: List[Dict], frame_width: int) -> float:
    """
    Calculate minimum distance between people in frame.
    
    Args:
        persons: List of person dicts with keypoints
        frame_width: Frame width for normalization
        
    Returns:
        Normalized distance (0-1) or -1 if < 2 people
    """
    if len(persons) < 2:
        return -1.0
    
    min_distance = float('inf')
    
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            # Use hip center as body center
            p1_kpts = persons[i].get('keypoints', [])
            p2_kpts = persons[j].get('keypoints', [])
            
            if len(p1_kpts) < 13 or len(p2_kpts) < 13:
                # Fallback to bbox center
                b1 = persons[i].get('bbox', [0, 0, 0, 0])
                b2 = persons[j].get('bbox', [0, 0, 0, 0])
                c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
            else:
                # Use hip midpoint
                c1 = ((p1_kpts[11][0] + p1_kpts[12][0]) / 2, 
                      (p1_kpts[11][1] + p1_kpts[12][1]) / 2)
                c2 = ((p2_kpts[11][0] + p2_kpts[12][0]) / 2,
                      (p2_kpts[11][1] + p2_kpts[12][1]) / 2)
            
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            min_distance = min(min_distance, dist)
    
    # Normalize by frame width
    return min(1.0, min_distance / frame_width) if frame_width > 0 else -1.0


def detect_interaction(persons: List[Dict], distance_threshold: float = 0.3) -> bool:
    """
    Detect if people are interacting based on pose and proximity.
    
    Args:
        persons: List of person dicts with keypoints and orientation
        distance_threshold: Max normalized distance for interaction
        
    Returns:
        True if interaction detected
    """
    if len(persons) < 2:
        return False
    
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            p1 = persons[i]
            p2 = persons[j]
            
            # Check if facing each other
            o1 = p1.get('body_orientation', 'unknown')
            o2 = p2.get('body_orientation', 'unknown')
            
            # Get positions
            b1 = p1.get('bbox', [0, 0, 0, 0])
            b2 = p2.get('bbox', [0, 0, 0, 0])
            c1_x = (b1[0] + b1[2]) / 2
            c2_x = (b2[0] + b2[2]) / 2
            
            facing_each_other = False
            
            # Person 1 is left of Person 2
            if c1_x < c2_x:
                if o1 == 'right' and o2 == 'left':
                    facing_each_other = True
                elif o1 == 'front' and o2 == 'front':
                    facing_each_other = True
            else:
                if o1 == 'left' and o2 == 'right':
                    facing_each_other = True
                elif o1 == 'front' and o2 == 'front':
                    facing_each_other = True
            
            if facing_each_other:
                return True
            
            # Also check for close proximity regardless of orientation
            # (e.g., one person behind another)
            kpts1 = p1.get('keypoints', [])
            kpts2 = p2.get('keypoints', [])
            
            if len(kpts1) >= 11 and len(kpts2) >= 11:
                # Check wrist proximity (touching/holding)
                for w1_idx in [9, 10]:  # wrists
                    for w2_idx in [9, 10]:
                        if kpts1[w1_idx][2] > 0.3 and kpts2[w2_idx][2] > 0.3:
                            dist = np.sqrt(
                                (kpts1[w1_idx][0] - kpts2[w2_idx][0])**2 +
                                (kpts1[w1_idx][1] - kpts2[w2_idx][1])**2
                            )
                            if dist < 50:  # Very close wrists
                                return True
    
    return False


# =============================================================================
# YOLO PERSON DETECTION + POSE ESTIMATION
# =============================================================================

class PersonDetector:
    """YOLO26x-based person detector with pose estimation."""
    
    def __init__(self, model_path: Path = YOLO_MODEL_PATH):
        """Initialize YOLO detection and pose models."""
        from ultralytics import YOLO
        
        self.model_path = model_path
        
        # Ensure weights directory exists
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Try to load YOLO26x models, fall back to YOLOv8 if not available
        try:
            if YOLO_DETECT_MODEL.exists():
                print(f"Loading YOLO26x detection model from {YOLO_DETECT_MODEL}...")
                self.detect_model = YOLO(str(YOLO_DETECT_MODEL))
            else:
                print(f"YOLO26x detection model not found at {YOLO_DETECT_MODEL}, downloading...")
                self.detect_model = YOLO('yolo26x.pt')
            
            if YOLO_POSE_MODEL.exists():
                print(f"Loading YOLO26x-pose model from {YOLO_POSE_MODEL}...")
                self.pose_model = YOLO(str(YOLO_POSE_MODEL))
            else:
                print(f"YOLO26x-pose model not found at {YOLO_POSE_MODEL}, downloading...")
                self.pose_model = YOLO('yolo26x-pose.pt')
            
            self.use_pose = True
            print("YOLO26x models loaded successfully")
        except Exception as e:
            print(f"YOLO26x not available ({e}), falling back to YOLOv8m")
            # Fallback to YOLOv8
            if model_path.exists():
                self.detect_model = YOLO(str(model_path))
            else:
                self.detect_model = YOLO('yolov8m.pt')
            
            # Try YOLOv8 pose as fallback
            try:
                self.pose_model = YOLO('yolov8m-pose.pt')
                self.use_pose = True
                print("Using YOLOv8m-pose for keypoint extraction")
            except:
                self.pose_model = None
                self.use_pose = False
                print("Pose estimation not available")
        
        # COCO class 0 is "person"
        self.person_class = 0
    
    def _load_detection_labels(self) -> list:
        """Load detection labels from configuration file."""
        config_path = Path(__file__).parent / "labels_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return config.get('nudenet', {}).get('detection_labels', [])
        return []  # Empty list means use all model labels
    
    def detect_in_frame(self, frame: np.ndarray) -> FrameDetection:
        """
        Run detection and pose estimation on a single frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            FrameDetection with results including pose data
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Run pose estimation if available (includes detection)
        if self.use_pose and self.pose_model is not None:
            results = self.pose_model(frame, verbose=False)
        else:
            # Fallback to detection only
            results = self.detect_model(frame, classes=[self.person_class], verbose=False)
        
        # Extract detections
        boxes = results[0].boxes if results else None
        keypoints_data = results[0].keypoints if (results and hasattr(results[0], 'keypoints')) else None
        
        if boxes is None or len(boxes) == 0:
            return FrameDetection(
                frame_idx=0,
                num_persons=0,
                max_confidence=0.0,
                total_bbox_area_ratio=0.0,
                persons=[],
                interpersonal_distance=-1.0,
                interaction_detected=False
            )
        
        # Process each detected person
        persons = []
        frame_area = frame_width * frame_height
        total_bbox_area = 0.0
        max_conf = 0.0
        
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < MIN_CONFIDENCE:
                continue
            
            max_conf = max(max_conf, conf)
            
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            bbox_area = (x2 - x1) * (y2 - y1)
            total_bbox_area += bbox_area
            
            # Get keypoints if available
            kpts = []
            body_orientation = 'unknown'
            posture = 'unknown'
            arm_position = 'unknown'
            
            if keypoints_data is not None and i < len(keypoints_data):
                try:
                    kpts_tensor = keypoints_data[i].data[0]  # Shape: (17, 3)
                    kpts_np = kpts_tensor.cpu().numpy()
                    kpts = kpts_np.tolist()
                    
                    # Analyze pose
                    body_orientation = analyze_body_orientation(kpts_np)
                    posture = analyze_posture(kpts_np)
                    arm_position = analyze_arm_position(kpts_np)
                except Exception as e:
                    pass
            
            person = {
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'keypoints': kpts,
                'body_orientation': body_orientation,
                'posture': posture,
                'arm_position': arm_position
            }
            persons.append(person)
        
        # Calculate interaction metrics
        interpersonal_dist = calculate_interpersonal_distance(persons, frame_width)
        interaction = detect_interaction(persons)
        
        return FrameDetection(
            frame_idx=0,
            num_persons=len(persons),
            max_confidence=max_conf,
            total_bbox_area_ratio=total_bbox_area / frame_area if frame_area > 0 else 0.0,
            persons=persons,
            interpersonal_distance=float(interpersonal_dist),
            interaction_detected=interaction
        )
    
    def detect_in_frame_legacy(self, frame: np.ndarray) -> FrameDetection:
        """
        Legacy detection method (detection only, no pose).
        Kept for backward compatibility.
        """
        # Run inference (only detect persons)
        results = self.detect_model(frame, classes=[self.person_class], verbose=False)
        
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
            max_confidence=float(max_conf),
            total_bbox_area_ratio=float(total_bbox_area / frame_area) if frame_area > 0 else 0.0
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
        import nudenet as nn
        
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
        # Supplement YOLOv8 detection with NudeNet 640 model for specific labels
        # We will log nudenet detections for EDA

        # ---- NudeNet Model Setup ----
        # Use the NudeDetector; load once per detector instance if possible
        if not hasattr(self, "_nudenet_model"):
            from nudenet import NudeDetector
            self._nudenet_model = NudeDetector()

        # Detection labels loaded from configuration (domain-specific)
        # Empty list means use all labels detected by the model
        all_labels = self._load_detection_labels()
        nudenet_log_path = Path("analysis") / "nudenet_scenes.json"
        if not hasattr(self, "_nudenet_frame_logs"):
            self._nudenet_frame_logs = {}

        # For each sampled frame, we'll run nudenet detection if YOLO fails or for EDA
        # We'll collect nudenet results for saving at script finish (EDA step)

        # Save the current video path log
        self._current_video_nudenet = []

        # Get video properties (ensure native Python types for JSON serialization)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        duration = float(total_frames / fps)
        
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
            float(np.mean([d.max_confidence for d in frames_with_person]))
            if frames_with_person else 0.0
        )
        
        avg_bbox_area = (
            float(np.mean([d.total_bbox_area_ratio for d in frames_with_person]))
            if frames_with_person else 0.0
        )
        
        max_persons = int(max(d.num_persons for d in detections))
        
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
    
    ROBUST VERSION with:
    - Incremental saving (crash-resistant)
    - Per-scene error handling
    - Resume capability
    
    Args:
        input_dir: Directory with preprocessed videos
        scenes_dir: Directory to save scene clips
        vlm_dir: Directory to save low-FPS VLM copies
        output_file: Path to save detection results JSON
        
    Returns:
        Summary statistics
    """
    from robust_processor import IncrementalJSONWriter
    
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
    
    # Initialize incremental writer
    writer = IncrementalJSONWriter(output_file, backup_interval=10)
    writer.set_config({
        'frame_sample_rate': FRAME_SAMPLE_RATE,
        'min_confidence': MIN_CONFIDENCE,
        'min_coverage': MIN_COVERAGE,
        'model_path': str(YOLO_MODEL_PATH),
        'low_fps_short': LOW_FPS_SHORT,
        'low_fps_long': LOW_FPS_LONG,
        'low_fps_threshold': LOW_FPS_THRESHOLD
    })
    
    # Get already processed for resume
    processed = writer.get_processed_paths()
    if processed:
        print(f"  Resuming: {len(processed)} scenes already processed")
    
    # Initialize detector
    print("Loading YOLO model...")
    detector = PersonDetector()
    
    # Track results
    total_scenes = len(processed)
    scenes_with_person = sum(1 for a in writer.data.get('analyses', []) if a.get('person_present', False))
    vlm_copies_created = 0
    failed = 0
    
    # Process each video
    for video_path in tqdm(videos, desc="Processing videos"):
        print(f"\n  Processing: {video_path.name}")
        
        # Step 1: Split into scenes
        try:
            print(f"    Splitting into scenes...")
            scene_files = split_video_into_scenes(video_path, scenes_dir)
            print(f"    Found {len(scene_files)} scenes")
        except Exception as e:
            print(f"    ⚠ Error splitting {video_path.name}: {e}")
            writer.add_error(str(video_path), 'scene_split_error', str(e))
            failed += 1
            continue
        
        if not scene_files:
            # If scene splitting fails, treat whole video as one scene
            scene_files = [video_path]
        
        # Step 2: Analyze each scene with error handling
        for scene_path in tqdm(scene_files, desc="    Analyzing scenes", leave=False):
            # Skip if already processed
            if str(scene_path) in processed:
                continue
            
            try:
                analysis = detector.analyze_video(scene_path)
                analysis_dict = asdict(analysis)
                writer.add_analysis(analysis_dict)
                total_scenes += 1
                
                if analysis.person_present:
                    scenes_with_person += 1
                    
                    # Step 3: Generate low-FPS copy for VLM
                    try:
                        vlm_path = generate_low_fps_copy(scene_path, vlm_dir)
                        if vlm_path:
                            vlm_copies_created += 1
                    except Exception as e:
                        print(f"\n      ⚠ VLM copy error: {e}")
                        # Continue anyway - VLM copy is not critical
                        
            except Exception as e:
                print(f"\n    ⚠ Error analyzing {scene_path.name}: {e}")
                writer.add_error(str(scene_path), type(e).__name__, str(e))
                failed += 1
                continue
    
    # Finalize
    summary = {
        'total_videos': len(videos),
        'total_scenes': total_scenes,
        'scenes_with_person': scenes_with_person,
        'scenes_without_person': total_scenes - scenes_with_person,
        'vlm_copies_created': vlm_copies_created,
        'failed': failed
    }
    
    writer.update_summary(summary)
    writer.finalize()
    
    print(f"\nResults saved to: {output_file}")
    print(f"  Successful: {total_scenes}, Failed: {failed}")
    
    return summary


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
