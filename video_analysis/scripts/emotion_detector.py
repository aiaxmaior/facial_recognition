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
from dataclasses import dataclass, asdict
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
class FrameEmotions:
    """Emotion analysis for a single frame."""
    frame_idx: int
    num_faces: int
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    valence: float  # Derived from emotion scores
    arousal: float  # Derived from emotion scores


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
    
    # Raw frame data (for detailed analysis)
    frame_emotions: List[Dict]


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
            
            # Aggregate emotions from all faces
            all_emotions = []
            for face_result in results:
                if 'emotion' in face_result:
                    all_emotions.append(face_result['emotion'])
            
            if not all_emotions:
                return None
            
            # Average emotion scores across all faces
            avg_emotions = {}
            for emotion in EMOTION_VALENCE.keys():
                scores = [e.get(emotion, 0) for e in all_emotions]
                avg_emotions[emotion] = np.mean(scores)
            
            # Normalize to sum to 100
            total = sum(avg_emotions.values())
            if total > 0:
                avg_emotions = {k: v/total * 100 for k, v in avg_emotions.items()}
            
            # Find dominant emotion
            dominant = max(avg_emotions, key=avg_emotions.get)
            
            # Calculate valence and arousal
            valence = self._calculate_valence(avg_emotions)
            arousal = self._calculate_arousal(avg_emotions)
            
            return FrameEmotions(
                frame_idx=frame_idx,
                num_faces=len(all_emotions),
                dominant_emotion=dominant,
                emotion_scores={k: round(v, 2) for k, v in avg_emotions.items()},
                valence=round(valence, 3),
                arousal=round(arousal, 3)
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
    Process all scenes with detected persons.
    
    Args:
        scenes_dir: Directory containing scene clips
        detections_file: Path to person detection results
        output_file: Path to save emotion analysis results
        
    Returns:
        Summary statistics
    """
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
    
    # Initialize analyzer
    print("Loading emotion analyzer...")
    analyzer = EmotionAnalyzer()
    
    # Process each scene
    all_analyses = []
    
    for scene_path in tqdm(scene_paths, desc="Analyzing emotions"):
        path = Path(scene_path)
        if not path.exists():
            continue
        
        analysis = analyzer.analyze_video(path)
        all_analyses.append(asdict(analysis))
    
    # Compile results
    results = {
        'config': {
            'sample_rate': EMOTION_SAMPLE_RATE,
            'valence_mapping': EMOTION_VALENCE,
            'arousal_mapping': EMOTION_AROUSAL
        },
        'summary': {
            'total_scenes': len(all_analyses),
            'scenes_with_faces': sum(1 for a in all_analyses if a['frames_with_faces'] > 0),
            'dominant_emotions': {
                e: sum(1 for a in all_analyses if a['dominant_emotion'] == e)
                for e in EMOTION_VALENCE.keys()
            }
        },
        'analyses': all_analyses
    }
    
    # Save results
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
