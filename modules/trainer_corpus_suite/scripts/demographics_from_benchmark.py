#!/usr/bin/env python3
"""
Demographics from Benchmark Results
====================================

Creates demographics.json from video benchmark results (video_benchmark_results.json).

This script:
1. Reads age/gender predictions from the benchmark (ViT-Age-Gender model)
2. Runs motion analysis on video clips
3. Outputs in the standard demographics.json format for the portal

Usage:
    python demographics_from_benchmark.py
    python demographics_from_benchmark.py --benchmark-path /path/to/video_benchmark_results.json
    python demographics_from_benchmark.py --skip-motion  # Skip motion analysis

Author: Pipeline Tools
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_BENCHMARK_PATH = SCRIPT_DIR.parent / "asdataset" / "video_benchmark_results" / "video_benchmark_results.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR.parent / "analysis" / "demographics.json"
DEFAULT_SCENES_DIR = SCRIPT_DIR.parent / "scenes"

# Motion analysis settings
ERRATIC_MOTION_THRESHOLD = 30.0

# Age categories
AGE_CATEGORIES = {
    'infant': (0, 2),
    'toddler': (2, 4),
    'child': (4, 9),
    'adolescent': (9, 13),
    'early_teen': (13, 16),
    'late_teen': (16, 19),
    'young_adult': (19, 30),
    'adult': (30, 45),
    'middle_aged': (45, 60),
    'senior': (60, 120),
}


def get_age_category(age: float) -> str:
    """Convert numeric age to category."""
    if age is None:
        return "unknown"
    for category, (min_age, max_age) in AGE_CATEGORIES.items():
        if min_age <= age < max_age:
            return category
    return "senior" if age >= 60 else "unknown"


# =============================================================================
# DATA STRUCTURES (matching demographics_detector.py format)
# =============================================================================

@dataclass
class PersonDemographics:
    """Demographics for a single detected person."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    # Age estimation
    age_mivolo: Optional[float] = None
    age_fairface: Optional[float] = None
    age_combined: Optional[float] = None
    age_category: str = "unknown"
    age_source: str = "vit_benchmark"
    
    # Gender estimation
    gender_mivolo: Optional[str] = None
    gender_fairface: Optional[str] = None
    gender_combined: Optional[str] = None
    gender_confidence: float = 0.0
    gender_source: str = "vit_benchmark"
    
    # Flags
    age_disagreement: bool = False
    gender_disagreement: bool = False
    needs_review: bool = False
    
    # Race (not available from benchmark)
    race: Optional[str] = None
    race_confidence: float = 0.0


@dataclass
class FrameDemographics:
    """Demographics for all persons in a frame."""
    frame_idx: int
    num_persons: int
    persons: List[Dict]
    
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
    
    # Race distribution
    race_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Motion analysis
    mean_motion: float = 0.0
    erratic_motion_ratio: float = 0.0
    
    # Raw frame data
    frame_demographics: List[Dict] = field(default_factory=list)


# =============================================================================
# MOTION ANALYZER (extracted from demographics_detector.py)
# =============================================================================

class MotionAnalyzer:
    """
    Optical flow-based motion analyzer.
    
    Detects erratic or unusual movement patterns.
    """
    
    def __init__(self):
        self.prev_gray = None
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """
        Analyze motion between current and previous frame.
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
        
        # Detect erratic motion
        angles = np.arctan2(flow[..., 1], flow[..., 0])
        moving_mask = magnitude > 1.0
        if np.any(moving_mask):
            angle_std = float(np.std(angles[moving_mask]))
        else:
            angle_std = 0.0
        
        erratic = (max_mag > ERRATIC_MOTION_THRESHOLD and angle_std > 1.5)
        
        self.prev_gray = gray
        
        return {
            'mean_magnitude': round(mean_mag, 2),
            'max_magnitude': round(max_mag, 2),
            'erratic_detected': erratic
        }
    
    def reset(self):
        """Reset for new video."""
        self.prev_gray = None


def analyze_video_motion(video_path: Path, sample_rate: int = 15) -> List[Dict]:
    """
    Analyze motion throughout a video.
    
    Returns list of motion metrics per sampled frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    analyzer = MotionAnalyzer()
    motion_data = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            motion = analyzer.analyze(frame)
            motion_data.append({
                'frame_idx': frame_idx,
                **motion
            })
        
        frame_idx += 1
    
    cap.release()
    return motion_data


# =============================================================================
# BENCHMARK RESULT PROCESSING
# =============================================================================

def load_benchmark_results(benchmark_path: Path) -> List[Dict]:
    """Load video benchmark results."""
    with open(benchmark_path) as f:
        return json.load(f)


def convert_benchmark_to_demographics(
    benchmark_data: List[Dict],
    scenes_dir: Path,
    skip_motion: bool = False,
    sample_rate: int = 15
) -> List[SceneDemographics]:
    """
    Convert benchmark results to demographics format.
    
    Args:
        benchmark_data: List of clip results from benchmark
        scenes_dir: Directory containing scene videos (for motion analysis)
        skip_motion: If True, skip motion analysis
        sample_rate: Frame sample rate for motion analysis
    
    Returns:
        List of SceneDemographics objects
    """
    results = []
    
    for clip_data in tqdm(benchmark_data, desc="Converting to demographics"):
        video_name = clip_data.get('video_name', '')
        video_path = clip_data.get('video_path', '')
        
        # Determine scene path
        if video_path:
            scene_path = video_path
        else:
            scene_path = str(scenes_dir / f"{video_name}.mp4")
        
        # Process frame predictions
        frame_predictions = clip_data.get('frame_predictions', [])
        frame_demographics = []
        all_ages = []
        all_genders = []
        frames_with_persons = 0
        
        # Motion analysis (if not skipped and video exists)
        motion_data = []
        if not skip_motion and Path(scene_path).exists():
            motion_data = analyze_video_motion(Path(scene_path), sample_rate)
        
        # Build motion lookup by frame_idx
        motion_lookup = {m['frame_idx']: m for m in motion_data}
        
        for frame_pred in frame_predictions:
            frame_idx = frame_pred.get('frame_idx', 0)
            predictions = frame_pred.get('predictions', {})
            
            # Get ViT prediction
            vit_pred = predictions.get('vit_age_gender', {})
            
            persons = []
            if vit_pred and vit_pred.get('age') is not None and vit_pred.get('error') is None:
                age = vit_pred['age']
                gender = vit_pred.get('gender', 'unknown')
                confidence = vit_pred.get('confidence', 0.8)
                
                # Create person entry (no bbox in benchmark, use full frame)
                height = clip_data.get('height', 544)
                width = clip_data.get('width', 960)
                
                person = PersonDemographics(
                    bbox=(0, 0, width, height),
                    age_combined=age,
                    age_category=get_age_category(age),
                    age_source='vit_benchmark',
                    gender_combined=gender,
                    gender_confidence=confidence,
                    gender_source='vit_benchmark'
                )
                persons.append(asdict(person))
                
                all_ages.append(age)
                all_genders.append(gender)
                frames_with_persons += 1
            
            # Get motion data for this frame
            motion = motion_lookup.get(frame_idx, {})
            
            frame_demo = FrameDemographics(
                frame_idx=frame_idx,
                num_persons=len(persons),
                persons=persons,
                mean_motion_magnitude=motion.get('mean_magnitude', 0.0),
                max_motion_magnitude=motion.get('max_magnitude', 0.0),
                erratic_motion_detected=motion.get('erratic_detected', False)
            )
            frame_demographics.append(asdict(frame_demo))
        
        # Build age distribution
        age_distribution = {cat: 0 for cat in AGE_CATEGORIES.keys()}
        age_distribution['unknown'] = 0
        for age in all_ages:
            cat = get_age_category(age)
            if cat in age_distribution:
                age_distribution[cat] += 1
        
        # Build gender distribution
        gender_distribution = {'Male': 0, 'Female': 0, 'unknown': 0}
        for gender in all_genders:
            g = gender.lower() if gender else 'unknown'
            if g in ('male', 'man'):
                gender_distribution['Male'] += 1
            elif g in ('female', 'woman'):
                gender_distribution['Female'] += 1
            else:
                gender_distribution['unknown'] += 1
        
        # Calculate aggregates
        mean_age = float(np.mean(all_ages)) if all_ages else 0.0
        age_range = (float(min(all_ages)), float(max(all_ages))) if all_ages else (0.0, 0.0)
        
        # Motion aggregates
        motion_mags = [m.get('mean_magnitude', 0) for m in motion_data]
        erratic_frames = sum(1 for m in motion_data if m.get('erratic_detected', False))
        mean_motion = float(np.mean(motion_mags)) if motion_mags else 0.0
        erratic_ratio = erratic_frames / len(motion_data) if motion_data else 0.0
        
        scene_demo = SceneDemographics(
            scene_path=scene_path,
            total_frames_analyzed=len(frame_demographics),
            frames_with_persons=frames_with_persons,
            age_distribution=age_distribution,
            mean_age=round(mean_age, 1),
            age_range=(round(age_range[0], 1), round(age_range[1], 1)),
            gender_distribution=gender_distribution,
            race_distribution={},
            mean_motion=round(mean_motion, 2),
            erratic_motion_ratio=round(erratic_ratio, 3),
            frame_demographics=frame_demographics
        )
        
        results.append(scene_demo)
    
    return results


def save_demographics(demographics: List[SceneDemographics], output_path: Path):
    """Save demographics in the standard format."""
    
    # Build summary
    total_age_dist = {cat: 0 for cat in list(AGE_CATEGORIES.keys()) + ['unknown']}
    total_gender_dist = {'Male': 0, 'Female': 0, 'unknown': 0}
    
    for demo in demographics:
        for cat, count in demo.age_distribution.items():
            if cat in total_age_dist:
                total_age_dist[cat] += count
        for gender, count in demo.gender_distribution.items():
            if gender in total_gender_dist:
                total_gender_dist[gender] += count
    
    output = {
        'config': {
            'sample_rate': 15,
            'age_categories': AGE_CATEGORIES,
            'erratic_motion_threshold': ERRATIC_MOTION_THRESHOLD,
            'source': 'video_benchmark (ViT-Age-Gender)'
        },
        'summary': {
            'total_scenes': len(demographics),
            'total_age_distribution': total_age_dist,
            'total_gender_distribution': total_gender_dist
        },
        'analyses': [asdict(d) for d in demographics]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved demographics to {output_path}")
    return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create demographics.json from video benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--benchmark-path', '-b', type=Path, default=DEFAULT_BENCHMARK_PATH,
                       help=f'Path to video_benchmark_results.json (default: {DEFAULT_BENCHMARK_PATH})')
    parser.add_argument('--output', '-o', type=Path, default=DEFAULT_OUTPUT_PATH,
                       help=f'Output demographics.json path (default: {DEFAULT_OUTPUT_PATH})')
    parser.add_argument('--scenes-dir', '-s', type=Path, default=DEFAULT_SCENES_DIR,
                       help=f'Directory with scene videos for motion analysis (default: {DEFAULT_SCENES_DIR})')
    parser.add_argument('--skip-motion', action='store_true',
                       help='Skip motion analysis (faster, but no motion data)')
    parser.add_argument('--sample-rate', type=int, default=15,
                       help='Frame sample rate for motion analysis (default: 15)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Demographics from Benchmark Results")
    print("=" * 70)
    print(f"Benchmark:   {args.benchmark_path}")
    print(f"Output:      {args.output}")
    print(f"Scenes Dir:  {args.scenes_dir}")
    print(f"Skip Motion: {args.skip_motion}")
    print("=" * 70)
    
    if not args.benchmark_path.exists():
        print(f"ERROR: Benchmark file not found: {args.benchmark_path}")
        return
    
    # Load benchmark results
    print("\nLoading benchmark results...")
    benchmark_data = load_benchmark_results(args.benchmark_path)
    print(f"  Found {len(benchmark_data)} clips")
    
    # Convert to demographics format
    print("\nConverting to demographics format...")
    demographics = convert_benchmark_to_demographics(
        benchmark_data,
        args.scenes_dir,
        skip_motion=args.skip_motion,
        sample_rate=args.sample_rate
    )
    
    # Save
    output = save_demographics(demographics, args.output)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Scenes: {output['summary']['total_scenes']}")
    
    print("\nAge Distribution:")
    for cat, count in sorted(output['summary']['total_age_distribution'].items()):
        if count > 0:
            print(f"  {cat}: {count}")
    
    print("\nGender Distribution:")
    for gender, count in output['summary']['total_gender_distribution'].items():
        if count > 0:
            print(f"  {gender}: {count}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
