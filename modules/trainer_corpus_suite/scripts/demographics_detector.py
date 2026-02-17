#!/usr/bin/env python3
"""
Demographics Detection Module
==============================

Multi-model demographic analysis using:
- YOLO26x: Person detection (upgraded from YOLOv8)
- MiVOLO v2: PRIMARY age/gender estimation (body + face aware)
- FairFace/DeepFace: SECONDARY age/gender/race validation
- Motion Analysis: Optical flow for erratic movement detection

Model Hierarchy:
- MiVOLO is the PRIMARY source for age/gender
- FairFace/DeepFace is used for validation and race detection
- Disagreements between models are FLAGGED for review

Optimized for RTX 3090 with concurrent CUDA streams.

CRITICAL: This script processes files WITHOUT interpreting visual content.
All operations are based on model outputs and statistical metrics only.

Usage:
    python demographics_detector.py [--scenes-dir DIR] [--output FILE]
"""

import json
import sys
import warnings
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import numpy as np
import cv2
import torch

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Frame sampling for demographics (more sparse - demographics don't change often)
DEMO_SAMPLE_RATE = 15  # Every 15th frame

# CUDA optimization settings
CUDA_STREAMS = 2  # Number of concurrent CUDA streams
BATCH_SIZE = 4    # Batch size for inference

# Weights directory (centralized for all models)
WEIGHTS_DIR = Path(__file__).parent / "weights"

# YOLO model (upgraded to YOLO26x)
YOLO_DETECT_MODEL = WEIGHTS_DIR / "yolo26x.pt"  # Will fallback to yolov8m if not available

# MiVOLO weights
MIVOLO_WEIGHTS = WEIGHTS_DIR / "model_imdb_cross_person_4.22_99.46.pth.tar"
MIVOLO_DETECTOR = WEIGHTS_DIR / "yolov8x_person_face.pt"  # Optional: MiVOLO's custom detector

# Age categories for binning
AGE_CATEGORIES = {
    'infant': (0, 3),
    'child': (3, 12),
    'adolescent': (12, 18),
    'young_adult': (18, 30),
    'adult': (30, 50),
    'middle_aged': (50, 65),
    'senior': (65, 100)
}

# Motion thresholds
ERRATIC_MOTION_THRESHOLD = 30.0  # Optical flow magnitude threshold

# Detection thresholds (increased to reduce false positives)
MIN_PERSON_CONFIDENCE = 0.5  # Minimum YOLO confidence for person detection (was 0.3)
MIN_PERSON_AREA_RATIO = 0.005  # Minimum bbox area as ratio of frame (filters tiny detections)
NMS_IOU_THRESHOLD = 0.5  # IoU threshold for person-level NMS (dedup overlapping boxes)

# Default paths
DEFAULT_SCENES_DIR = Path(__file__).parent.parent / "scenes"
DEFAULT_DETECTIONS_FILE = Path(__file__).parent.parent / "analysis" / "detections.json"
DEFAULT_OUTPUT_FILE = Path(__file__).parent.parent / "analysis" / "demographics.json"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PersonDemographics:
    """Demographics for a single detected person."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    # Age estimation (MiVOLO is PRIMARY)
    age_mivolo: Optional[float] = None
    age_fairface: Optional[float] = None
    age_combined: Optional[float] = None
    age_category: str = "unknown"
    age_source: str = "unknown"  # 'mivolo', 'fairface', 'combined'
    
    # Gender estimation (MiVOLO is PRIMARY)
    gender_mivolo: Optional[str] = None
    gender_fairface: Optional[str] = None
    gender_combined: Optional[str] = None
    gender_confidence: float = 0.0
    gender_source: str = "unknown"  # 'mivolo', 'fairface', 'combined'
    
    # DISAGREEMENT FLAGS
    age_disagreement: bool = False      # MiVOLO and FairFace disagree on age (>10 years diff)
    gender_disagreement: bool = False   # MiVOLO and FairFace disagree on gender
    needs_review: bool = False          # Flagged for manual review
    
    # Race/ethnicity (FairFace only)
    race: Optional[str] = None
    race_confidence: float = 0.0


@dataclass
class FrameDemographics:
    """Demographics for all persons in a frame."""
    frame_idx: int
    num_persons: int
    persons: List[Dict]  # Serialized PersonDemographics
    
    # Motion metrics
    mean_motion_magnitude: float = 0.0
    max_motion_magnitude: float = 0.0
    erratic_motion_detected: bool = False


@dataclass  
class SceneDemographics:
    """Aggregated demographics for a scene."""
    scene_path: str
    total_frames_analyzed: int
    frames_with_persons: int
    
    # Aggregated age distribution
    age_distribution: Dict[str, int] = field(default_factory=dict)
    mean_age: float = 0.0
    age_range: Tuple[float, float] = (0.0, 0.0)
    
    # Aggregated gender distribution  
    gender_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Race distribution (if detected)
    race_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Motion analysis
    mean_motion: float = 0.0
    erratic_motion_ratio: float = 0.0  # % of frames with erratic motion
    
    # Raw frame data
    frame_demographics: List[Dict] = field(default_factory=list)


# =============================================================================
# MIVOLO AGE/GENDER DETECTOR
# =============================================================================

class MiVOLODetector:
    """
    MiVOLO v2 age/gender detector - PRIMARY demographics source.
    
    Uses body-aware detection for improved accuracy even with partial faces.
    Falls back to face-only when body not visible.
    
    Model: https://github.com/WildChlamydia/MiVOLO
    """
    
    # Age tolerance for agreement (years)
    AGE_AGREEMENT_THRESHOLD = 10
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.detector = None
        self.use_fallback = False
        self._load_models()
    
    def _load_models(self):
        """Load MiVOLO models."""
        try:
            # Try to import mivolo package
            from mivolo.predictor import Predictor
            from mivolo.model.create_timm_model import create_model
            
            # Ensure weights directory exists
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Check for MiVOLO weights
            model_path = MIVOLO_WEIGHTS
            detector_path = MIVOLO_DETECTOR
            
            if not model_path.exists():
                print(f"MiVOLO: Weights not found at {model_path}")
                print("MiVOLO: Please download weights from https://github.com/WildChlamydia/MiVOLO")
                raise FileNotFoundError(f"MiVOLO weights not found: {model_path}")
            
            # Use YOLO26x or fallback detector
            if not detector_path.exists():
                # Try to use YOLO26x as person detector instead
                if YOLO_DETECT_MODEL.exists():
                    detector_path = YOLO_DETECT_MODEL
                    print(f"MiVOLO: Using YOLO26x for person detection: {detector_path}")
                else:
                    print(f"MiVOLO: Detector not found at {detector_path}, downloading yolov8x...")
                    from ultralytics import YOLO
                    YOLO('yolov8x.pt')  # Download default
                    detector_path = Path('yolov8x.pt')
            
            # Initialize predictor
            print(f"MiVOLO: Loading model from {model_path}")
            self.predictor = Predictor(
                detector_weights=str(detector_path),
                checkpoint=str(model_path),
                device=self.device,
                with_persons=True,
                disable_faces=False
            )
            self.use_fallback = False
            print("MiVOLO: Loaded successfully (PRIMARY age/gender source)")
            
        except ImportError:
            print("MiVOLO: Package not installed, trying pip install...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "mivolo", "-q"])
                # Retry after install
                self._load_models()
            except Exception as e:
                print(f"MiVOLO: Could not install package: {e}")
                self._setup_fallback()
                
        except Exception as e:
            print(f"MiVOLO: Load failed ({e}), using YOLO fallback")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback using YOLO + heuristics."""
        try:
            from ultralytics import YOLO
            self.detector = YOLO('yolov8m.pt')
            self.use_fallback = True
            print("MiVOLO: Using YOLO detection with DeepFace fallback for age/gender")
        except Exception as e:
            print(f"MiVOLO: Fallback setup failed: {e}")
            self.detector = None
            self.use_fallback = True
    
    def predict(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> Dict:
        """
        Predict age and gender for a person in image.
        
        Args:
            image: Full frame (BGR numpy array)
            bbox: Optional person bounding box (x1, y1, x2, y2)
        
        Returns:
            Dict with 'age', 'gender', 'confidence' keys
        """
        result = {'age': None, 'gender': None, 'confidence': 0.0}
        
        if self.use_fallback:
            return result  # Let FairFace handle it
        
        try:
            # Run MiVOLO predictor on full image
            detected_objects, _ = self.predictor.recognize(image)
            
            if detected_objects is None or detected_objects.n_objects == 0:
                return result
            
            # Get person indices from the result
            person_inds = detected_objects.get_bboxes_inds("person")
            face_inds = detected_objects.get_bboxes_inds("face")
            
            # If bbox provided, find matching detection
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                best_idx = None
                best_iou = 0
                
                # Check persons first (more reliable for body detection)
                for idx in person_inds:
                    obj_bbox = detected_objects.get_bbox_by_ind(idx).cpu().numpy()
                    iou = self._calculate_iou(bbox, tuple(obj_bbox))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                
                # Also check faces
                for idx in face_inds:
                    obj_bbox = detected_objects.get_bbox_by_ind(idx).cpu().numpy()
                    iou = self._calculate_iou(bbox, tuple(obj_bbox))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                
                if best_idx is not None and best_iou > 0.1:
                    age = detected_objects.ages[best_idx]
                    gender = detected_objects.genders[best_idx]
                    if age is not None:
                        result['age'] = float(age)
                        result['gender'] = 'Male' if gender == 'male' else 'Female' if gender == 'female' else gender
                        result['confidence'] = 0.8
            else:
                # Return first person/face with age
                for idx in person_inds + face_inds:
                    age = detected_objects.ages[idx]
                    gender = detected_objects.genders[idx]
                    if age is not None:
                        result['age'] = float(age)
                        result['gender'] = 'Male' if gender == 'male' else 'Female' if gender == 'female' else gender
                        result['confidence'] = 0.8
                        break
                        
        except Exception as e:
            # Only log if we haven't seen this error recently (reduce spam)
            err_msg = str(e) if str(e) else type(e).__name__
            if not hasattr(self, '_last_error') or self._last_error != err_msg:
                self._last_error = err_msg
                print(f"  MiVOLO predict error: {err_msg}")
        
        return result
    
    def _calculate_iou(self, box1, box2) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


# =============================================================================
# MIVOLO API CLIENT (FALLBACK)
# =============================================================================

class MiVOLOAPIClient:
    """
    MiVOLO API client - calls external MiVOLO service.
    
    Use this when local MiVOLO model fails to load or for distributed processing.
    API endpoint: http://127.0.0.1:5005/analyze
    
    Key features:
    - Analyzes FULL FRAME even without bounding boxes (API does its own detection)
    - Falls back gracefully if API unavailable
    - Can detect persons that YOLO might have missed
    """
    
    DEFAULT_API_URL = "http://127.0.0.1:5005/analyze"
    TIMEOUT = 30  # seconds
    
    def __init__(self, api_url: str = None, device: str = "cuda"):
        self.api_url = api_url or self.DEFAULT_API_URL
        self.device = device  # Not used but kept for interface compatibility
        self.use_fallback = False
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if API is available."""
        try:
            # Try a simple request with a tiny test image
            test_img = np.zeros((10, 10, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', test_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                self.api_url,
                json={"image_base64": img_base64},
                timeout=5
            )
            if response.status_code == 200:
                print(f"MiVOLO API: Connected to {self.api_url}")
                return True
        except Exception as e:
            print(f"MiVOLO API: Not available at {self.api_url} ({e})")
        return False
    
    def predict(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> Dict:
        """
        Predict age and gender via API.
        
        Args:
            image: Full frame (BGR numpy array)
            bbox: Optional person bounding box (x1, y1, x2, y2)
                  If None, API analyzes the full frame
        
        Returns:
            Dict with 'age', 'gender', 'confidence' keys
        """
        result = {'age': None, 'gender': None, 'confidence': 0.0}
        
        if not self.available:
            return result
        
        try:
            # Optionally crop to bounding box, or send full frame
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Add padding around bbox
                h, w = image.shape[:2]
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    cropped = image
            else:
                # Send full frame - API will do its own detection
                cropped = image
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Call API
            response = requests.post(
                self.api_url,
                json={"image_base64": img_base64},
                timeout=self.TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse response - handle different formats
                if 'analysis' in data:
                    analysis = data['analysis']
                    
                    # Check if it's "No people detected" message
                    if isinstance(analysis, str) and "No people" in analysis:
                        return result
                    
                    # Try to parse structured response
                    if isinstance(analysis, dict):
                        result['age'] = analysis.get('age')
                        result['gender'] = analysis.get('gender')
                        result['confidence'] = analysis.get('confidence', 0.8)
                    elif isinstance(analysis, list) and len(analysis) > 0:
                        # Take first person
                        person = analysis[0]
                        result['age'] = person.get('age')
                        result['gender'] = person.get('gender')
                        result['confidence'] = person.get('confidence', 0.8)
                
                # Also check for direct keys
                if result['age'] is None and 'age' in data:
                    result['age'] = data.get('age')
                    result['gender'] = data.get('gender')
                    result['confidence'] = data.get('confidence', 0.8)
                    
                # Parse people array if present
                if result['age'] is None and 'people' in data and len(data['people']) > 0:
                    person = data['people'][0]
                    result['age'] = person.get('age')
                    result['gender'] = person.get('gender')
                    result['confidence'] = person.get('confidence', 0.8)
                        
        except requests.Timeout:
            print(f"  MiVOLO API timeout")
        except Exception as e:
            print(f"  MiVOLO API error: {e}")
        
        return result
    
    def predict_frame(self, image: np.ndarray) -> List[Dict]:
        """
        Analyze full frame and return ALL detected persons.
        
        This is useful when YOLO didn't detect anyone but there might still be people.
        
        Returns:
            List of dicts with 'age', 'gender', 'confidence', 'bbox' for each person
        """
        results = []
        
        if not self.available:
            return results
            
        try:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                self.api_url,
                json={"image_base64": img_base64},
                timeout=self.TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse all detected people
                if 'people' in data:
                    for person in data['people']:
                        results.append({
                            'age': person.get('age'),
                            'gender': person.get('gender'),
                            'confidence': person.get('confidence', 0.8),
                            'bbox': person.get('bbox')
                        })
                elif 'analysis' in data and isinstance(data['analysis'], list):
                    for person in data['analysis']:
                        results.append({
                            'age': person.get('age'),
                            'gender': person.get('gender'),
                            'confidence': person.get('confidence', 0.8),
                            'bbox': person.get('bbox')
                        })
                        
        except Exception as e:
            pass  # Silently fail for batch operations
            
        return results


# =============================================================================
# FAIRFACE AGE/GENDER/RACE DETECTOR
# =============================================================================

class FairFaceDetector:
    """
    FairFace attribute detector.
    
    Designed for reduced bias across demographics.
    Predicts: age, gender, race
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load FairFace model."""
        try:
            # FairFace uses ResNet-34 backbone
            import torchvision.transforms as transforms
            from torchvision import models
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # For now, use DeepFace as fallback which includes similar functionality
            try:
                from deepface import DeepFace
                self.deepface = DeepFace
                self.use_deepface = True
                print("FairFace: Using DeepFace backend for demographics")
            except ImportError:
                self.deepface = None
                self.use_deepface = False
                print("FairFace: DeepFace not available")
            
        except Exception as e:
            print(f"FairFace load failed: {e}")
            self.model = None
            self.use_deepface = False
    
    def predict(self, face_image: np.ndarray) -> Dict:
        """
        Predict age, gender, race from face image.
        
        Args:
            face_image: BGR face crop
            
        Returns:
            Dict with 'age', 'gender', 'race' keys
        """
        if self.use_deepface and self.deepface is not None:
            try:
                # Use DeepFace analyze
                result = self.deepface.analyze(
                    face_image,
                    actions=['age', 'gender', 'race'],
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                return {
                    'age': result.get('age'),
                    'gender': result.get('dominant_gender'),
                    'gender_confidence': max(result.get('gender', {}).values()) / 100 if result.get('gender') else 0,
                    'race': result.get('dominant_race'),
                    'race_confidence': max(result.get('race', {}).values()) / 100 if result.get('race') else 0
                }
                
            except Exception as e:
                pass
        
        return {
            'age': None,
            'gender': None,
            'gender_confidence': 0,
            'race': None,
            'race_confidence': 0
        }


# =============================================================================
# MOTION ANALYSIS (ERRATIC MOVEMENT DETECTION)
# =============================================================================

class MotionAnalyzer:
    """
    Optical flow-based motion analyzer.
    
    Detects erratic or unusual movement patterns that may indicate
    specific types of activity or emotional states.
    """
    
    def __init__(self):
        self.prev_frame = None
        self.prev_gray = None
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """
        Analyze motion between current and previous frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Dict with motion metrics
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return {
                'mean_magnitude': 0.0,
                'max_magnitude': 0.0,
                'erratic_detected': False
            }
        
        # Calculate optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Calculate flow magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        mean_mag = float(np.mean(magnitude))
        max_mag = float(np.max(magnitude))
        
        # Detect erratic motion (high variance in flow direction)
        angles = np.arctan2(flow[..., 1], flow[..., 0])
        angle_std = float(np.std(angles[magnitude > 1.0]))  # Only consider moving pixels
        
        # Erratic = high magnitude + high directional variance
        erratic = (max_mag > ERRATIC_MOTION_THRESHOLD and angle_std > 1.5)
        
        self.prev_gray = gray
        
        return {
            'mean_magnitude': round(mean_mag, 2),
            'max_magnitude': round(max_mag, 2),
            'erratic_detected': erratic,
            'direction_variance': round(angle_std, 3)
        }
    
    def reset(self):
        """Reset motion state for new video."""
        self.prev_frame = None
        self.prev_gray = None


# =============================================================================
# COMBINED DEMOGRAPHICS ANALYZER
# =============================================================================

class DemographicsAnalyzer:
    """
    Combined demographics analyzer using multiple models.
    
    Optimized for RTX 3090 with CUDA stream concurrency.
    """
    
    def __init__(self, device: str = None, use_api: bool = False, api_url: str = None):
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_api = use_api
        
        print(f"Initializing demographics analyzer on {device}")
        
        # Initialize MiVOLO (local model or API)
        if use_api:
            print("Using MiVOLO API for demographics...")
            self.mivolo = MiVOLOAPIClient(api_url=api_url, device=device)
            self.mivolo_api = self.mivolo  # Keep reference for full-frame analysis
        else:
            self.mivolo = MiVOLODetector(device)
            # Also try to initialize API as fallback
            self.mivolo_api = MiVOLOAPIClient(api_url=api_url, device=device)
            if not self.mivolo_api.available:
                self.mivolo_api = None
        
        self.fairface = FairFaceDetector(device)
        self.motion = MotionAnalyzer()
        
        # Face detector for FairFace
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
        except ImportError:
            self.deepface = None
        
        # YOLO for person detection (try YOLO26x, fallback to YOLOv8m)
        from ultralytics import YOLO
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        
        if YOLO_DETECT_MODEL.exists():
            self.person_detector = YOLO(str(YOLO_DETECT_MODEL))
            print(f"Using YOLO26x for person detection: {YOLO_DETECT_MODEL}")
        else:
            print(f"YOLO26x not found at {YOLO_DETECT_MODEL}, downloading/using default...")
            try:
                self.person_detector = YOLO('yolo26x.pt')
            except:
                print("Falling back to yolov8m.pt")
                self.person_detector = YOLO('yolov8m.pt')
        
        print("Demographics analyzer initialized")
    
    def _get_age_category(self, age: float) -> str:
        """Convert numeric age to category."""
        if age is None:
            return "unknown"
        
        for category, (min_age, max_age) in AGE_CATEGORIES.items():
            if min_age <= age < max_age:
                return category
        return "unknown"
    
    def _apply_person_nms(self, boxes_with_conf: List[Tuple], iou_threshold: float = NMS_IOU_THRESHOLD) -> List[Tuple]:
        """
        Apply Non-Maximum Suppression to person detections to remove duplicates.
        
        Args:
            boxes_with_conf: List of (bbox, confidence) tuples
            iou_threshold: IoU threshold above which boxes are considered duplicates
            
        Returns:
            Filtered list of (bbox, confidence) tuples
        """
        if not boxes_with_conf:
            return []
        
        # Sort by confidence (highest first)
        sorted_boxes = sorted(boxes_with_conf, key=lambda x: x[1], reverse=True)
        
        keep = []
        while sorted_boxes:
            best = sorted_boxes.pop(0)
            keep.append(best)
            
            # Filter remaining boxes that overlap too much with best
            remaining = []
            for other in sorted_boxes:
                iou = self._calculate_iou(best[0], other[0])
                if iou < iou_threshold:
                    remaining.append(other)
            sorted_boxes = remaining
        
        return keep
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union between two boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> FrameDemographics:
        """
        Analyze demographics in a single frame.
        
        Args:
            frame: BGR image
            frame_idx: Frame index
            
        Returns:
            FrameDemographics with all detected persons
        """
        # Detect persons
        results = self.person_detector(frame, classes=[0], verbose=False)
        boxes = results[0].boxes if results else None
        
        persons = []
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        
        if boxes is not None and len(boxes) > 0:
            # First pass: filter by confidence and size, collect candidates
            candidates = []
            for box in boxes:
                conf = float(box.conf[0])
                if conf < MIN_PERSON_CONFIDENCE:  # Use configurable threshold (was 0.3)
                    continue
                
                # Get bbox
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                bbox = (x1, y1, x2, y2)
                
                # Filter tiny detections (noise)
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area / frame_area < MIN_PERSON_AREA_RATIO:
                    continue
                
                candidates.append((bbox, conf))
            
            # Second pass: apply person-level NMS to remove duplicates
            filtered_candidates = self._apply_person_nms(candidates, NMS_IOU_THRESHOLD)
            
            # Third pass: process filtered candidates
            for bbox, conf in filtered_candidates:
                x1, y1, x2, y2 = bbox
                
                # Crop person region
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                
                # Get demographics from MiVOLO (PRIMARY source)
                mv_result = self.mivolo.predict(frame, bbox)
                
                # Get demographics from FairFace/DeepFace (SECONDARY/validation)
                ff_result = self.fairface.predict(person_crop)
                
                # Extract values
                age_mv = mv_result.get('age')
                age_ff = ff_result.get('age')
                gender_mv = mv_result.get('gender')
                gender_ff = ff_result.get('gender')
                mv_conf = mv_result.get('confidence', 0)
                ff_conf = ff_result.get('gender_confidence', 0)
                
                # === AGE DETERMINATION (MiVOLO primary) ===
                age_disagreement = False
                if age_mv is not None:
                    age_combined = age_mv
                    age_source = 'mivolo'
                    # Check for disagreement with FairFace
                    if age_ff is not None:
                        age_diff = abs(age_mv - age_ff)
                        if age_diff > 10:  # More than 10 years difference
                            age_disagreement = True
                            # Use weighted average when disagreement
                            age_combined = (age_mv * 0.6 + age_ff * 0.4)
                            age_source = 'combined'
                elif age_ff is not None:
                    age_combined = age_ff
                    age_source = 'fairface'
                else:
                    age_combined = None
                    age_source = 'unknown'
                
                # === GENDER DETERMINATION (MiVOLO primary) ===
                gender_disagreement = False
                if gender_mv is not None:
                    gender_combined = gender_mv
                    gender_source = 'mivolo'
                    gender_conf = mv_conf
                    # Check for disagreement with FairFace
                    if gender_ff is not None and gender_mv != gender_ff:
                        gender_disagreement = True
                        # Use higher confidence when disagreement
                        if ff_conf > mv_conf:
                            gender_combined = gender_ff
                            gender_source = 'fairface'
                            gender_conf = ff_conf
                elif gender_ff is not None:
                    gender_combined = gender_ff
                    gender_source = 'fairface'
                    gender_conf = ff_conf
                else:
                    gender_combined = None
                    gender_source = 'unknown'
                    gender_conf = 0
                
                # Flag for review if any disagreement
                needs_review = age_disagreement or gender_disagreement
                
                person = PersonDemographics(
                    bbox=bbox,
                    age_mivolo=age_mv,
                    age_fairface=age_ff,
                    age_combined=age_combined,
                    age_category=self._get_age_category(age_combined),
                    age_source=age_source,
                    gender_mivolo=gender_mv,
                    gender_fairface=gender_ff,
                    gender_combined=gender_combined,
                    gender_confidence=float(gender_conf),
                    gender_source=gender_source,
                    age_disagreement=age_disagreement,
                    gender_disagreement=gender_disagreement,
                    needs_review=needs_review,
                    race=ff_result.get('race'),
                    race_confidence=ff_result.get('race_confidence', 0)
                )
                
                persons.append(asdict(person))
        
        # === FALLBACK: Try MiVOLO API for full-frame when YOLO found nothing ===
        # This catches cases where YOLO misses but MiVOLO's detector finds people
        if len(persons) == 0 and self.mivolo_api is not None and self.mivolo_api.available:
            try:
                api_persons = self.mivolo_api.predict_frame(frame)
                for api_person in api_persons:
                    if api_person.get('age') is not None:
                        # Create demographics from API result only
                        age = api_person.get('age')
                        gender = api_person.get('gender')
                        conf = api_person.get('confidence', 0.8)
                        bbox = api_person.get('bbox') or (0, 0, frame.shape[1], frame.shape[0])
                        
                        person = PersonDemographics(
                            bbox=bbox,
                            age_mivolo=age,
                            age_fairface=None,
                            age_combined=age,
                            age_category=self._get_age_category(age),
                            age_source='mivolo_api',
                            gender_mivolo=gender,
                            gender_fairface=None,
                            gender_combined=gender,
                            gender_confidence=float(conf),
                            gender_source='mivolo_api',
                            age_disagreement=False,
                            gender_disagreement=False,
                            needs_review=False,  # API-only results marked for potential review
                            race=None,
                            race_confidence=0
                        )
                        persons.append(asdict(person))
            except Exception as e:
                pass  # Silently continue if API fallback fails
        
        # Analyze motion
        motion = self.motion.analyze(frame)
        
        return FrameDemographics(
            frame_idx=frame_idx,
            num_persons=len(persons),
            persons=persons,
            mean_motion_magnitude=motion['mean_magnitude'],
            max_motion_magnitude=motion['max_magnitude'],
            erratic_motion_detected=motion['erratic_detected']
        )
    
    def analyze_video(self, video_path: Path, 
                      sample_rate: int = DEMO_SAMPLE_RATE) -> SceneDemographics:
        """
        Analyze demographics throughout a video.
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            
        Returns:
            SceneDemographics with aggregated results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return SceneDemographics(
                scene_path=str(video_path),
                total_frames_analyzed=0,
                frames_with_persons=0
            )
        
        # Reset motion analyzer
        self.motion.reset()
        
        frame_demographics = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                demo = self.analyze_frame(frame, frame_idx)
                frame_demographics.append(demo)
            
            frame_idx += 1
        
        cap.release()
        
        return self._aggregate_demographics(video_path, frame_demographics)
    
    def _aggregate_demographics(self, video_path: Path,
                                frame_demographics: List[FrameDemographics]) -> SceneDemographics:
        """Aggregate frame-level demographics into scene-level."""
        
        if not frame_demographics:
            return SceneDemographics(
                scene_path=str(video_path),
                total_frames_analyzed=0,
                frames_with_persons=0
            )
        
        # Initialize aggregation
        all_ages = []
        age_dist = {cat: 0 for cat in AGE_CATEGORIES.keys()}
        age_dist['unknown'] = 0
        gender_dist = {'Male': 0, 'Female': 0, 'unknown': 0}
        race_dist = {}
        
        frames_with_persons = 0
        erratic_frames = 0
        motion_mags = []
        
        for fd in frame_demographics:
            if fd.num_persons > 0:
                frames_with_persons += 1
            
            if fd.erratic_motion_detected:
                erratic_frames += 1
            
            motion_mags.append(fd.mean_motion_magnitude)
            
            for person in fd.persons:
                # Age
                age = person.get('age_combined')
                if age is not None:
                    all_ages.append(age)
                
                age_cat = person.get('age_category', 'unknown')
                if age_cat in age_dist:
                    age_dist[age_cat] += 1
                
                # Gender
                gender = person.get('gender_fairface') or person.get('gender_mivolo')
                if gender in gender_dist:
                    gender_dist[gender] += 1
                else:
                    gender_dist['unknown'] += 1
                
                # Race
                race = person.get('race')
                if race:
                    race_dist[race] = race_dist.get(race, 0) + 1
        
        # Calculate aggregates
        mean_age = float(np.mean(all_ages)) if all_ages else 0.0
        age_range = (float(min(all_ages)), float(max(all_ages))) if all_ages else (0.0, 0.0)
        mean_motion = float(np.mean(motion_mags)) if motion_mags else 0.0
        erratic_ratio = erratic_frames / len(frame_demographics) if frame_demographics else 0.0
        
        return SceneDemographics(
            scene_path=str(video_path),
            total_frames_analyzed=len(frame_demographics),
            frames_with_persons=frames_with_persons,
            age_distribution=age_dist,
            mean_age=round(mean_age, 1),
            age_range=(round(age_range[0], 1), round(age_range[1], 1)),
            gender_distribution=gender_dist,
            race_distribution=race_dist,
            mean_motion=round(mean_motion, 2),
            erratic_motion_ratio=round(erratic_ratio, 3),
            frame_demographics=[asdict(fd) for fd in frame_demographics]
        )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def load_detection_results(detections_file: Path) -> Dict:
    """Load person detection results."""
    with open(detections_file) as f:
        return json.load(f)


def get_scenes_with_persons(detections: Dict) -> List[str]:
    """Extract paths of scenes where persons were detected."""
    return [
        a['scene_path'] 
        for a in detections.get('analyses', [])
        if a.get('person_present', False)
    ]


def process_scenes(scenes_dir: Path, 
                   detections_file: Path,
                   output_file: Path,
                   use_api: bool = False,
                   api_url: str = None,
                   device: str = None) -> Dict:
    """
    Process all scenes with detected persons - ROBUST VERSION.
    
    Features:
    - Incremental saving (crash-resistant)
    - Error handling per-scene (doesn't break on single failure)
    - Timeout handling for stuck videos
    - Resume capability
    
    Args:
        scenes_dir: Directory containing scene clips
        detections_file: Path to person detection results
        output_file: Path to save demographics analysis results
        
    Returns:
        Summary statistics
    """
    from robust_processor import IncrementalJSONWriter, process_scene_robust
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load detection results
    print("Loading detection results...")
    
    if detections_file.exists():
        detections = load_detection_results(detections_file)
        scene_paths = get_scenes_with_persons(detections)
        print(f"Found {len(scene_paths)} scenes with persons detected")
    else:
        # If no detection file, process all videos in scenes dir
        print(f"No detection file found, processing all scenes in {scenes_dir}")
        scene_paths = [str(p) for p in sorted(scenes_dir.glob("*.mp4"))]
        print(f"Found {len(scene_paths)} scene files")
    
    if not scene_paths:
        print("No scenes to process")
        return {'total_scenes': 0}
    
    # Initialize incremental writer (supports resume)
    writer = IncrementalJSONWriter(output_file, backup_interval=5)
    writer.set_config({
        'sample_rate': DEMO_SAMPLE_RATE,
        'age_categories': AGE_CATEGORIES,
        'erratic_motion_threshold': ERRATIC_MOTION_THRESHOLD
    })
    
    # Get already processed for resume
    processed = writer.get_processed_paths()
    if processed:
        print(f"  Resuming: {len(processed)} scenes already processed")
    
    # Initialize analyzer
    print("Loading demographics analyzer...")
    analyzer = DemographicsAnalyzer(device=device, use_api=use_api, api_url=api_url)
    
    # Define processing function for single scene
    def analyze_single(scene_path: Path) -> Dict:
        analysis = analyzer.analyze_video(scene_path)
        return asdict(analysis)
    
    # Process each scene with error handling
    successful = 0
    failed = 0
    skipped = 0
    
    for scene_path_str in tqdm(scene_paths, desc="Analyzing demographics"):
        scene_path = Path(scene_path_str)
        
        # Skip if already processed (resume)
        if str(scene_path) in processed:
            skipped += 1
            continue
        
        # Process with timeout and error handling
        result = process_scene_robust(
            scene_path,
            analyze_single,
            timeout_seconds=180  # 3 min timeout per scene (demographics is slower)
        )
        
        if result.success:
            writer.add_analysis(result.data)
            successful += 1
        else:
            writer.add_error(result.scene_path, result.error_type, result.error_message)
            failed += 1
            print(f"\n  ⚠ Error on {scene_path.name}: {result.error_type}")
    
    # Finalize and compute summary
    all_analyses = writer.data.get('analyses', [])
    
    # Aggregate statistics
    total_age_dist = {cat: 0 for cat in list(AGE_CATEGORIES.keys()) + ['unknown']}
    total_gender_dist = {'Male': 0, 'Female': 0, 'unknown': 0}
    
    for a in all_analyses:
        for cat, count in a.get('age_distribution', {}).items():
            if cat in total_age_dist:
                total_age_dist[cat] += count
        for gender, count in a.get('gender_distribution', {}).items():
            if gender in total_gender_dist:
                total_gender_dist[gender] += count
    
    summary = {
        'total_scenes': len(all_analyses),
        'successful': successful,
        'failed': failed,
        'skipped_resume': skipped,
        'total_age_distribution': total_age_dist,
        'total_gender_distribution': total_gender_dist
    }
    
    writer.update_summary(summary)
    writer.finalize()
    
    print(f"\nResults saved to: {output_file}")
    print(f"  Successful: {successful}, Failed: {failed}, Resumed: {skipped}")
    
    return summary


def _legacy_save_results(results, output_file):
    """Legacy save function - kept for reference."""
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
        description="Analyze demographics in scene videos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--scenes-dir', '-s',
        type=Path,
        default=DEFAULT_SCENES_DIR,
        help=f"Directory with scene videos (default: {DEFAULT_SCENES_DIR})"
    )
    
    parser.add_argument(
        '--detections', '-d',
        type=Path,
        default=DEFAULT_DETECTIONS_FILE,
        help=f"Person detection results file (default: {DEFAULT_DETECTIONS_FILE})"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})"
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    parser.add_argument(
        '--use-api',
        action='store_true',
        help="Use MiVOLO API at 127.0.0.1:5005 instead of local model"
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default="http://127.0.0.1:5005/analyze",
        help="MiVOLO API URL (default: http://127.0.0.1:5005/analyze)"
    )
    
    args = parser.parse_args()
    
    # Print config
    print("=" * 60)
    print("Demographics Detection")
    print("=" * 60)
    print(f"Scenes Dir:  {args.scenes_dir}")
    print(f"Detections:  {args.detections}")
    print(f"Output:      {args.output}")
    print(f"Sample Rate: Every {DEMO_SAMPLE_RATE} frames")
    print(f"Device:      {args.device or 'auto'}")
    if args.use_api:
        print(f"MiVOLO:      API mode ({args.api_url})")
    else:
        print(f"MiVOLO:      Local model (API fallback: {args.api_url})")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Process
    summary = process_scenes(
        args.scenes_dir, args.detections, args.output,
        use_api=args.use_api, api_url=args.api_url, device=args.device
    )
    
    # Report
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Scenes: {summary.get('total_scenes', 0)}")
    
    if 'total_age_distribution' in summary:
        print("\nAge Distribution:")
        for cat, count in sorted(summary['total_age_distribution'].items()):
            if count > 0:
                print(f"  {cat}: {count}")
    
    if 'total_gender_distribution' in summary:
        print("\nGender Distribution:")
        for gender, count in summary['total_gender_distribution'].items():
            if count > 0:
                print(f"  {gender}: {count}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
