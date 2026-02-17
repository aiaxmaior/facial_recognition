#!/usr/bin/env python3
"""
Emotion Detection Module
========================

Analyzes scenes for facial emotions using DeepFace.

CRITICAL: This script processes files WITHOUT interpreting visual content.
All operations are based on model outputs and statistical metrics only.

Outputs:
- 7 basic emotions: angry, disgust, fear, happy, sad, surprise, neutral
- Derived Valence/Arousal estimates from emotion distributions

The emotions are used to derive pain/pleasure indicators:
- Pain indicators: sad, fear, angry, disgust
- Pleasure indicators: happy, surprise (positive)
- Neutral: baseline

Usage:
    python emotion_detector.py [--input-dir DIR] [--detections-file FILE]
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import numpy as np

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# CONFIGURATION
# =============================================================================

# Frame sampling for emotion detection (more sparse than person detection)
EMOTION_SAMPLE_RATE = 10  # Every 10th frame

# Minimum face confidence for emotion analysis
MIN_FACE_CONFIDENCE = 0.5

# Valence mapping for emotions (positive = pleasure, negative = pain)
# Scale: -1 (extreme pain) to +1 (extreme pleasure)
EMOTION_VALENCE = {
    'angry': -0.6,
    'disgust': -0.7,
    'fear': -0.8,
    'happy': 0.9,
    'sad': -0.7,
    'surprise': 0.3,  # Can be positive or negative, slight positive bias
    'neutral': 0.0
}

# Arousal mapping (activation level)
# Scale: 0 (calm) to 1 (highly activated)
EMOTION_AROUSAL = {
    'angry': 0.8,
    'disgust': 0.5,
    'fear': 0.9,
    'happy': 0.7,
    'sad': 0.3,
    'surprise': 0.9,
    'neutral': 0.2
}

# Default paths
DEFAULT_SCENES_DIR = Path(__file__).parent.parent / "scenes"
DEFAULT_DETECTIONS_FILE = Path(__file__).parent.parent / "analysis" / "detections.json"
DEFAULT_OUTPUT_FILE = Path(__file__).parent.parent / "analysis" / "emotions.json"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FaceEmotions:
    """Emotion analysis for a single detected face."""
    face_idx: int
    bbox: Optional[Tuple[int, int, int, int]]  # x1, y1, x2, y2 if available
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    valence: float
    arousal: float
    confidence: float


@dataclass
class FrameEmotions:
    """Emotion analysis for a single frame."""
    frame_idx: int
    num_faces: int
    dominant_emotion: str  # Dominant across all faces (for backwards compat)
    emotion_scores: Dict[str, float]  # Averaged scores (for backwards compat)
    valence: float  # Average valence (for backwards compat)
    arousal: float  # Average arousal (for backwards compat)
    # NEW: Per-face emotion data
    faces: List[Dict] = None  # List of FaceEmotions as dicts


@dataclass
class SceneEmotionAnalysis:
    """Aggregated emotion analysis for a scene."""
    scene_path: str
    total_frames_analyzed: int
    frames_with_faces: int
    
    # Dominant emotion across scene
    dominant_emotion: str
    emotion_distribution: Dict[str, float]  # Average scores
    
    # Valence/Arousal (derived from emotions)
    mean_valence: float
    mean_arousal: float
    valence_range: Tuple[float, float]
    arousal_range: Tuple[float, float]
    
    # Pain/Pleasure derived score
    # positive = pleasure dominant, negative = pain dominant
    pain_pleasure_score: float
    
    # NEW: Entropy - measures emotion uncertainty/ambiguity
    # Low entropy = clear dominant emotion, High entropy = mixed emotions
    emotion_entropy: float = 0.0
    
    # NEW: Temporal dynamics
    valence_volatility: float = 0.0   # Std dev of valence (stability measure)
    arousal_volatility: float = 0.0   # Std dev of arousal
    valence_trend: float = 0.0        # Slope of valence over time (-1 to +1)
    arousal_trend: float = 0.0        # Slope of arousal over time
    
    # NEW: Emotional arc classification
    # 'escalating' | 'de-escalating' | 'stable' | 'volatile' | 'building' | 'releasing'
    emotional_arc: str = 'stable'
    
    # Raw frame data (for detailed analysis)
    frame_emotions: List[Dict] = field(default_factory=list)


# =============================================================================
# ENTROPY AND TEMPORAL DYNAMICS FUNCTIONS
# =============================================================================

def calculate_emotion_entropy(emotion_scores: Dict[str, float]) -> float:
    """
    Calculate Shannon entropy of emotion distribution.
    
    Low entropy (0-1): Clear dominant emotion, high certainty
    High entropy (>2): Mixed/ambiguous emotions, low certainty
    Max entropy for 7 emotions: log2(7) ≈ 2.81
    
    Args:
        emotion_scores: Dict of emotion -> percentage (0-100)
        
    Returns:
        Entropy value (0 to ~2.81 for 7 emotions)
    """
    # Convert percentages to probabilities (0-1)
    total = sum(emotion_scores.values())
    if total == 0:
        return 0.0
    
    probs = np.array([score / total for score in emotion_scores.values()])
    
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    # Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))
    return round(float(entropy), 3)


def calculate_trend(values: List[float]) -> float:
    """
    Calculate linear trend (slope) of values over time.
    
    Args:
        values: List of values (ordered by time/frame)
        
    Returns:
        Normalized slope (-1 to +1 range)
    """
    if len(values) < 2:
        return 0.0
    
    # Use linear regression to find slope
    x = np.arange(len(values))
    try:
        slope, _ = np.polyfit(x, values, 1)
        
        # Normalize to -1 to +1 range based on value range
        value_range = max(values) - min(values)
        if value_range > 0:
            # Normalize slope relative to total change over sequence
            normalized_slope = slope * len(values) / value_range
            # Clip to -1 to +1
            normalized_slope = max(-1, min(1, normalized_slope))
        else:
            normalized_slope = 0.0
        
        return round(float(normalized_slope), 3)
    except:
        return 0.0


def classify_emotional_arc(
    valence_trend: float,
    arousal_trend: float,
    valence_volatility: float,
    arousal_volatility: float
) -> str:
    """
    Classify the emotional arc of a scene based on temporal dynamics.
    
    Returns one of:
    - 'escalating': Increasing intensity (arousal up, valence up)
    - 'de-escalating': Decreasing intensity (arousal down)
    - 'building': Tension building (arousal up, valence down)
    - 'releasing': Tension release (arousal down, valence up)
    - 'volatile': High variability, unstable emotions
    - 'stable': Minimal change throughout
    
    Args:
        valence_trend: Valence slope (-1 to +1)
        arousal_trend: Arousal slope (-1 to +1)
        valence_volatility: Std dev of valence
        arousal_volatility: Std dev of arousal
        
    Returns:
        Emotional arc classification string
    """
    # High volatility = volatile emotions
    if valence_volatility > 0.3 or arousal_volatility > 0.3:
        return 'volatile'
    
    # Minimal trends = stable
    if abs(valence_trend) < 0.2 and abs(arousal_trend) < 0.2:
        return 'stable'
    
    # Classify based on trend directions
    v_up = valence_trend > 0.2
    v_down = valence_trend < -0.2
    a_up = arousal_trend > 0.2
    a_down = arousal_trend < -0.2
    
    if a_up and v_up:
        return 'escalating'
    elif a_down and v_down:
        return 'de-escalating'
    elif a_up and v_down:
        return 'building'
    elif a_down and v_up:
        return 'releasing'
    elif a_up:
        return 'intensifying'
    elif a_down:
        return 'calming'
    elif v_up:
        return 'improving'
    elif v_down:
        return 'declining'
    
    return 'stable'


# =============================================================================
# EMOTION ANALYSIS
# =============================================================================

class EmotionAnalyzer:
    """DeepFace-based emotion analyzer."""
    
    def __init__(self):
        """Initialize DeepFace (lazy loading)."""
        self.deepface = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load DeepFace to avoid startup delay."""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            print("DeepFace loaded successfully")
        except ImportError:
            print("Warning: DeepFace not available, emotion analysis will be skipped")
            self.deepface = None
    
    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[FrameEmotions]:
        """
        Analyze emotions in a single frame.
        
        Args:
            frame: BGR image as numpy array
            frame_idx: Frame index
            
        Returns:
            FrameEmotions or None if no faces detected
        """
        if self.deepface is None:
            return None
        
        try:
            # Run DeepFace analysis
            results = self.deepface.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if not results:
                return None
            
            # Handle both single face and multiple faces
            if isinstance(results, dict):
                results = [results]
            
            # === NEW: Store per-face emotion data ===
            faces_data = []
            all_emotions = []
            
            for face_idx, face_result in enumerate(results):
                if 'emotion' not in face_result:
                    continue
                    
                face_emotions = face_result['emotion']
                all_emotions.append(face_emotions)
                
                # Normalize this face's emotions to sum to 100
                total = sum(face_emotions.values())
                if total > 0:
                    normalized_emotions = {k: v/total * 100 for k, v in face_emotions.items()}
                else:
                    normalized_emotions = face_emotions
                
                # Find this face's dominant emotion
                face_dominant = max(normalized_emotions, key=normalized_emotions.get)
                
                # Calculate this face's valence and arousal
                face_valence = self._calculate_valence(normalized_emotions)
                face_arousal = self._calculate_arousal(normalized_emotions)
                
                # Extract bbox if available (DeepFace provides 'region')
                region = face_result.get('region', {})
                bbox = None
                if region:
                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                    if w > 0 and h > 0:
                        bbox = (x, y, x + w, y + h)
                
                # Store per-face data
                face_data = {
                    'face_idx': face_idx,
                    'bbox': bbox,
                    'dominant_emotion': face_dominant,
                    'emotion_scores': {k: round(v, 2) for k, v in normalized_emotions.items()},
                    'valence': round(face_valence, 3),
                    'arousal': round(face_arousal, 3),
                    'confidence': face_result.get('face_confidence', 0.0)
                }
                faces_data.append(face_data)
            
            if not all_emotions:
                return None
            
            # === Aggregate for backwards compatibility ===
            avg_emotions = {}
            for emotion in EMOTION_VALENCE.keys():
                scores = [e.get(emotion, 0) for e in all_emotions]
                avg_emotions[emotion] = np.mean(scores)
            
            # Normalize to sum to 100
            total = sum(avg_emotions.values())
            if total > 0:
                avg_emotions = {k: v/total * 100 for k, v in avg_emotions.items()}
            
            # Find dominant emotion (aggregate)
            dominant = max(avg_emotions, key=avg_emotions.get)
            
            # Calculate valence and arousal (aggregate)
            valence = self._calculate_valence(avg_emotions)
            arousal = self._calculate_arousal(avg_emotions)
            
            return FrameEmotions(
                frame_idx=frame_idx,
                num_faces=len(all_emotions),
                dominant_emotion=dominant,
                emotion_scores={k: round(v, 2) for k, v in avg_emotions.items()},
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                faces=faces_data  # NEW: Per-face data
            )
            
        except Exception as e:
            # Silently skip problematic frames
            return None
    
    def _calculate_valence(self, emotions: Dict[str, float]) -> float:
        """
        Calculate valence from emotion distribution.
        
        Valence = weighted sum of emotion scores * emotion valence values
        """
        valence = 0.0
        for emotion, score in emotions.items():
            weight = score / 100  # Convert percentage to 0-1
            valence += weight * EMOTION_VALENCE.get(emotion, 0)
        return valence
    
    def _calculate_arousal(self, emotions: Dict[str, float]) -> float:
        """
        Calculate arousal from emotion distribution.
        
        Arousal = weighted sum of emotion scores * emotion arousal values
        """
        arousal = 0.0
        for emotion, score in emotions.items():
            weight = score / 100
            arousal += weight * EMOTION_AROUSAL.get(emotion, 0)
        return arousal
    
    def analyze_video(self, video_path: Path, 
                      sample_rate: int = EMOTION_SAMPLE_RATE) -> SceneEmotionAnalysis:
        """
        Analyze emotions throughout a video.
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            
        Returns:
            SceneEmotionAnalysis with aggregated results
        """
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return self._empty_analysis(video_path)
        
        frame_emotions = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                emotions = self.analyze_frame(frame, frame_idx)
                if emotions:
                    frame_emotions.append(emotions)
            
            frame_idx += 1
        
        cap.release()
        
        return self._aggregate_emotions(video_path, frame_emotions, frame_idx // sample_rate)
    
    def _empty_analysis(self, video_path: Path) -> SceneEmotionAnalysis:
        """Return empty analysis for videos with no detections."""
        return SceneEmotionAnalysis(
            scene_path=str(video_path),
            total_frames_analyzed=0,
            frames_with_faces=0,
            dominant_emotion='neutral',
            emotion_distribution={e: 0 for e in EMOTION_VALENCE.keys()},
            mean_valence=0,
            mean_arousal=0,
            valence_range=(0, 0),
            arousal_range=(0, 0),
            pain_pleasure_score=0,
            emotion_entropy=0.0,
            valence_volatility=0.0,
            arousal_volatility=0.0,
            valence_trend=0.0,
            arousal_trend=0.0,
            emotional_arc='stable',
            frame_emotions=[]
        )
    
    def _aggregate_emotions(self, video_path: Path, 
                           frame_emotions: List[FrameEmotions],
                           total_analyzed: int) -> SceneEmotionAnalysis:
        """Aggregate frame-level emotions into scene-level analysis."""
        
        if not frame_emotions:
            return self._empty_analysis(video_path)
        
        # Aggregate emotion scores
        avg_emotions = {e: 0 for e in EMOTION_VALENCE.keys()}
        for fe in frame_emotions:
            for emotion, score in fe.emotion_scores.items():
                avg_emotions[emotion] += score
        
        # Average
        n = len(frame_emotions)
        avg_emotions = {k: round(v/n, 2) for k, v in avg_emotions.items()}
        
        # Find dominant emotion
        dominant = max(avg_emotions, key=avg_emotions.get)
        
        # Valence/arousal statistics
        valences = [fe.valence for fe in frame_emotions]
        arousals = [fe.arousal for fe in frame_emotions]
        
        mean_valence = np.mean(valences)
        mean_arousal = np.mean(arousals)
        
        # Pain/pleasure score (weighted valence with arousal as intensity)
        # Higher arousal amplifies the valence
        pain_pleasure = mean_valence * (0.5 + 0.5 * mean_arousal)
        
        # NEW: Calculate emotion entropy (uncertainty/ambiguity)
        entropy = calculate_emotion_entropy(avg_emotions)
        
        # NEW: Calculate temporal dynamics (volatility and trends)
        valence_volatility = round(float(np.std(valences)), 3) if len(valences) > 1 else 0.0
        arousal_volatility = round(float(np.std(arousals)), 3) if len(arousals) > 1 else 0.0
        
        valence_trend = calculate_trend(valences)
        arousal_trend = calculate_trend(arousals)
        
        # NEW: Classify emotional arc
        emotional_arc = classify_emotional_arc(
            valence_trend, arousal_trend,
            valence_volatility, arousal_volatility
        )
        
        return SceneEmotionAnalysis(
            scene_path=str(video_path),
            total_frames_analyzed=total_analyzed,
            frames_with_faces=n,
            dominant_emotion=dominant,
            emotion_distribution=avg_emotions,
            mean_valence=round(mean_valence, 3),
            mean_arousal=round(mean_arousal, 3),
            valence_range=(round(min(valences), 3), round(max(valences), 3)),
            arousal_range=(round(min(arousals), 3), round(max(arousals), 3)),
            pain_pleasure_score=round(pain_pleasure, 3),
            emotion_entropy=entropy,
            valence_volatility=valence_volatility,
            arousal_volatility=arousal_volatility,
            valence_trend=valence_trend,
            arousal_trend=arousal_trend,
            emotional_arc=emotional_arc,
            frame_emotions=[asdict(fe) for fe in frame_emotions]
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
                   output_file: Path) -> Dict:
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
        output_file: Path to save emotion analysis results
        
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
        'sample_rate': EMOTION_SAMPLE_RATE,
        'valence_mapping': EMOTION_VALENCE,
        'arousal_mapping': EMOTION_AROUSAL
    })
    
    # Get already processed for resume
    processed = writer.get_processed_paths()
    if processed:
        print(f"  Resuming: {len(processed)} scenes already processed")
    
    # Initialize analyzer
    print("Loading emotion analyzer...")
    analyzer = EmotionAnalyzer()
    
    # Define processing function for single scene
    def analyze_single(scene_path: Path) -> Dict:
        analysis = analyzer.analyze_video(scene_path)
        return asdict(analysis)
    
    # Process each scene with error handling
    successful = 0
    failed = 0
    skipped = 0
    
    for scene_path_str in tqdm(scene_paths, desc="Analyzing emotions"):
        scene_path = Path(scene_path_str)
        
        # Skip if already processed (resume)
        if str(scene_path) in processed:
            skipped += 1
            continue
        
        # Process with timeout and error handling
        result = process_scene_robust(
            scene_path,
            analyze_single,
            timeout_seconds=120  # 2 min timeout per scene
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
    
    summary = {
        'total_scenes': len(all_analyses),
        'scenes_with_faces': sum(1 for a in all_analyses if a.get('frames_with_faces', 0) > 0),
        'successful': successful,
        'failed': failed,
        'skipped_resume': skipped,
        'dominant_emotions': {
            e: sum(1 for a in all_analyses if a.get('dominant_emotion') == e)
            for e in EMOTION_VALENCE.keys()
        }
    }
    
    writer.update_summary(summary)
    writer.finalize()
    
    print(f"\nResults saved to: {output_file}")
    print(f"  Successful: {successful}, Failed: {failed}, Resumed: {skipped}")
    
    return summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze emotions in scene videos",
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
    
    args = parser.parse_args()
    
    # Print config
    print("=" * 60)
    print("Emotion Detection")
    print("=" * 60)
    print(f"Scenes Dir:  {args.scenes_dir}")
    print(f"Detections:  {args.detections}")
    print(f"Output:      {args.output}")
    print(f"Sample Rate: Every {EMOTION_SAMPLE_RATE} frames")
    print("=" * 60)
    
    # Process
    summary = process_scenes(args.scenes_dir, args.detections, args.output)
    
    # Report
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Scenes:     {summary.get('total_scenes', 0)}")
    print(f"Scenes w/ Faces:  {summary.get('scenes_with_faces', 0)}")
    
    if 'dominant_emotions' in summary:
        print("\nDominant Emotions:")
        for emotion, count in sorted(summary['dominant_emotions'].items(), 
                                     key=lambda x: -x[1]):
            if count > 0:
                print(f"  {emotion}: {count}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
