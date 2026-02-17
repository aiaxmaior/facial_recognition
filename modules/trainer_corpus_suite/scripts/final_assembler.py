#!/usr/bin/env python3
"""
Final Dataset Assembler
=======================

Combines all detector outputs and VLM captions into a unified dataset.
Output includes full labels for user review and downstream processing.

Creates:
1. unified_dataset.csv - Complete dataset with all metadata
2. training_manifest.json - Structured format for model training

Output columns include:
- Scene identification and paths
- YOLO person detection metrics
- Emotion analysis (valence/arousal/dominant emotion)
- Demographics (age/gender/motion)
- NudeNet detections
- VLM caption
- [BEST STILL] timestamp (from VLM)
- Composite priority scores

CRITICAL: This output contains FULL LABELS and is FOR USER REVIEW ONLY.
The AI assistant should NOT read this file.

Usage:
    python final_assembler.py [--output-csv FILE] [--output-json FILE]
"""

import json
import csv
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
DETECTIONS_FILE = Path(__file__).parent.parent / "analysis" / "detections.json"
EMOTIONS_FILE = Path(__file__).parent.parent / "analysis" / "emotions.json"
DEMOGRAPHICS_FILE = Path(__file__).parent.parent / "analysis" / "demographics.json"
NUDENET_FILE = Path(__file__).parent.parent / "analysis" / "nudenet.json"
CAPTIONS_FILE = Path(__file__).parent.parent / "analysis" / "captions.json"

# Output files
DEFAULT_CSV_OUTPUT = Path(__file__).parent.parent / "curated" / "unified_dataset.csv"
DEFAULT_JSON_OUTPUT = Path(__file__).parent.parent / "curated" / "training_manifest.json"

# Scene directories
SCENES_DIR = Path(__file__).parent.parent / "scenes"
VLM_DIR = Path(__file__).parent.parent / "vlm_copies"

# Priority scoring weights
PRIORITY_WEIGHTS = {
    'person_coverage': 0.15,
    'emotional_intensity': 0.25,  # |valence| + arousal
    'demographic_clarity': 0.15,
    'nudenet_relevance': 0.20,
    'caption_quality': 0.15,
    'motion_interest': 0.10
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class UnifiedSceneRecord:
    """Complete record for a single scene with all metadata."""
    # Identification
    scene_name: str
    scene_path: str
    vlm_path: str
    source_video: str
    
    # YOLO Detection
    person_present: bool
    person_coverage: float
    person_confidence: float
    max_persons: int
    
    # Emotion Analysis
    dominant_emotion: str
    emotion_confidence: float
    valence: float
    arousal: float
    emotional_intensity: float  # Derived: |valence| + arousal
    
    # Demographics
    age_estimates: str  # Comma-separated
    gender_estimates: str  # Comma-separated
    motion_intensity: float
    
    # NudeNet
    nudenet_labels: str  # Comma-separated with counts
    nudenet_detection_rate: float
    has_exposed_content: bool
    
    # VLM Caption
    caption: str
    caption_length: int
    best_still_timestamp: Optional[float]  # Extracted from caption
    vlm_validation_notes: str  # Any validation/rejection from VLM
    
    # Computed Scores
    priority_score: float
    training_suitability: str  # 'high', 'medium', 'low'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json_safe(file_path: Path) -> Dict:
    """Load JSON file if it exists."""
    if file_path.exists():
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    return {}


def load_all_data() -> Dict[str, Dict]:
    """Load all detector outputs."""
    return {
        'detections': load_json_safe(DETECTIONS_FILE),
        'emotions': load_json_safe(EMOTIONS_FILE),
        'demographics': load_json_safe(DEMOGRAPHICS_FILE),
        'nudenet': load_json_safe(NUDENET_FILE),
        'captions': load_json_safe(CAPTIONS_FILE)
    }


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_best_still_timestamp(caption: str) -> Optional[float]:
    """
    Extract [BEST STILL] timestamp from VLM caption.
    
    Looks for patterns like:
    - [BEST STILL]: 2.5s
    - [BEST STILL] timestamp: 00:02.5
    - Best still frame: 2.5 seconds
    
    Returns timestamp in seconds or None if not found.
    """
    if not caption:
        return None
    
    patterns = [
        r'\[BEST STILL\][:\s]*(\d+\.?\d*)\s*s',
        r'\[BEST STILL\][:\s]*(\d+):(\d+\.?\d*)',
        r'best still[:\s]*(\d+\.?\d*)\s*(?:s|sec)',
        r'ideal frame[:\s]*(\d+\.?\d*)\s*(?:s|sec)',
        r'timestamp[:\s]*(\d+\.?\d*)\s*(?:s|sec)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, caption, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:  # MM:SS format
                return float(groups[0]) * 60 + float(groups[1])
            else:
                return float(groups[0])
    
    return None


def extract_validation_notes(caption: str) -> str:
    """
    Extract any validation/rejection notes from VLM caption.
    
    Looks for sections where VLM validates or rejects detector data.
    """
    if not caption:
        return ""
    
    # Look for validation-related keywords
    validation_keywords = [
        'validates', 'confirms', 'accurate', 'correct',
        'rejects', 'incorrect', 'inaccurate', 'wrong',
        'partially correct', 'adjustment needed'
    ]
    
    notes = []
    sentences = caption.split('.')
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for keyword in validation_keywords:
            if keyword in sentence_lower:
                notes.append(sentence.strip())
                break
    
    return ' | '.join(notes[:3])  # Limit to 3 notes


def get_detection_data(scene_name: str, detections: Dict) -> Dict:
    """Extract detection data for a scene."""
    default = {
        'person_present': False,
        'person_coverage': 0.0,
        'person_confidence': 0.0,
        'max_persons': 0
    }
    
    if not detections or 'analyses' not in detections:
        return default
    
    for analysis in detections['analyses']:
        if scene_name in analysis.get('scene_path', ''):
            return {
                'person_present': analysis.get('person_present', False),
                'person_coverage': analysis.get('detection_coverage', 0.0),
                'person_confidence': analysis.get('avg_confidence', 0.0),
                'max_persons': analysis.get('max_persons_detected', 0)
            }
    
    return default


def get_emotion_data(scene_name: str, emotions: Dict) -> Dict:
    """Extract emotion data for a scene."""
    default = {
        'dominant_emotion': 'unknown',
        'emotion_confidence': 0.0,
        'valence': 0.0,
        'arousal': 0.0
    }
    
    if not emotions or 'analyses' not in emotions:
        return default
    
    for analysis in emotions['analyses']:
        if scene_name in analysis.get('scene_path', ''):
            return {
                'dominant_emotion': analysis.get('dominant_emotion', 'unknown'),
                'emotion_confidence': analysis.get('emotion_confidence', 0.0),
                'valence': analysis.get('mean_valence', 0.0),
                'arousal': analysis.get('mean_arousal', 0.0)
            }
    
    return default


def get_demographics_data(scene_name: str, demographics: Dict) -> Dict:
    """Extract demographics data for a scene."""
    default = {
        'age_estimates': '',
        'gender_estimates': '',
        'motion_intensity': 0.0
    }
    
    if not demographics or 'analyses' not in demographics:
        return default
    
    for analysis in demographics['analyses']:
        if scene_name in analysis.get('scene_path', ''):
            ages = analysis.get('age_categories', [])
            genders = analysis.get('genders', [])
            return {
                'age_estimates': ', '.join(ages) if ages else '',
                'gender_estimates': ', '.join(genders) if genders else '',
                'motion_intensity': analysis.get('motion_intensity', 0.0)
            }
    
    return default


def get_nudenet_data(scene_name: str, nudenet: Dict) -> Dict:
    """Extract NudeNet data for a scene."""
    default = {
        'nudenet_labels': '',
        'nudenet_detection_rate': 0.0,
        'has_exposed_content': False
    }
    
    if not nudenet or 'analyses' not in nudenet:
        return default
    
    scene_data = nudenet['analyses'].get(scene_name, {})
    label_counts = scene_data.get('label_counts', {})
    
    # Format labels with counts
    labels_str = ', '.join([f"{k}:{v}" for k, v in sorted(label_counts.items(), key=lambda x: -x[1])])
    
    # Check for exposed content
    exposed_labels = ['EXPOSED' in k for k in label_counts.keys()]
    has_exposed = any(exposed_labels)
    
    return {
        'nudenet_labels': labels_str,
        'nudenet_detection_rate': scene_data.get('detection_rate', 0.0),
        'has_exposed_content': has_exposed
    }


def get_caption_data(scene_name: str, captions: Dict) -> Dict:
    """Extract caption data for a scene."""
    default = {
        'caption': '',
        'caption_length': 0,
        'best_still_timestamp': None,
        'vlm_validation_notes': ''
    }
    
    if not captions or 'results' not in captions:
        return default
    
    for result in captions['results']:
        if result.get('scene_name') == scene_name:
            caption = result.get('caption', '')
            return {
                'caption': caption,
                'caption_length': len(caption) if caption else 0,
                'best_still_timestamp': extract_best_still_timestamp(caption),
                'vlm_validation_notes': extract_validation_notes(caption)
            }
    
    return default


# =============================================================================
# PRIORITY SCORING
# =============================================================================

def compute_priority_score(record: UnifiedSceneRecord) -> float:
    """
    Compute priority score for training suitability.
    
    Higher scores = more suitable for training.
    """
    scores = {}
    
    # Person coverage (0-1)
    scores['person_coverage'] = record.person_coverage
    
    # Emotional intensity (|valence| + arousal, normalized to 0-1)
    emotional = (abs(record.valence) + record.arousal) / 2
    scores['emotional_intensity'] = min(emotional, 1.0)
    
    # Demographic clarity (has both age and gender)
    has_age = 1.0 if record.age_estimates else 0.0
    has_gender = 1.0 if record.gender_estimates else 0.0
    scores['demographic_clarity'] = (has_age + has_gender) / 2
    
    # NudeNet relevance (detection rate)
    scores['nudenet_relevance'] = record.nudenet_detection_rate
    
    # Caption quality (based on length, normalized)
    caption_score = min(record.caption_length / 500, 1.0)  # 500 chars = max
    scores['caption_quality'] = caption_score
    
    # Motion interest (higher motion = more dynamic)
    scores['motion_interest'] = min(record.motion_intensity / 50, 1.0)  # 50 = max
    
    # Weighted sum
    total = sum(
        scores[k] * PRIORITY_WEIGHTS[k]
        for k in PRIORITY_WEIGHTS
    )
    
    return round(total, 3)


def get_training_suitability(score: float) -> str:
    """Categorize training suitability based on score."""
    if score >= 0.6:
        return 'high'
    elif score >= 0.3:
        return 'medium'
    else:
        return 'low'


# =============================================================================
# ASSEMBLY
# =============================================================================

def discover_scenes() -> List[str]:
    """Discover all scene names from the scenes directory."""
    if not SCENES_DIR.exists():
        return []
    
    scenes = set()
    for f in SCENES_DIR.glob("*.mp4"):
        # Extract base scene name (without extension)
        scene_name = f.stem
        scenes.add(scene_name)
    
    return sorted(scenes)


def assemble_record(scene_name: str, data: Dict) -> UnifiedSceneRecord:
    """Assemble complete record for a scene."""
    # Get paths
    scene_path = SCENES_DIR / f"{scene_name}.mp4"
    vlm_path = VLM_DIR / f"{scene_name}_vlm.mp4"
    
    # Extract source video from scene name (format: videoname-Scene-NNN)
    parts = scene_name.rsplit('-Scene-', 1)
    source_video = parts[0] if len(parts) > 1 else scene_name
    
    # Get data from each detector
    det_data = get_detection_data(scene_name, data['detections'])
    emo_data = get_emotion_data(scene_name, data['emotions'])
    demo_data = get_demographics_data(scene_name, data['demographics'])
    nn_data = get_nudenet_data(scene_name, data['nudenet'])
    cap_data = get_caption_data(scene_name, data['captions'])
    
    # Calculate emotional intensity
    emotional_intensity = abs(emo_data['valence']) + emo_data['arousal']
    
    # Create record
    record = UnifiedSceneRecord(
        scene_name=scene_name,
        scene_path=str(scene_path) if scene_path.exists() else '',
        vlm_path=str(vlm_path) if vlm_path.exists() else '',
        source_video=source_video,
        
        person_present=det_data['person_present'],
        person_coverage=det_data['person_coverage'],
        person_confidence=det_data['person_confidence'],
        max_persons=det_data['max_persons'],
        
        dominant_emotion=emo_data['dominant_emotion'],
        emotion_confidence=emo_data['emotion_confidence'],
        valence=emo_data['valence'],
        arousal=emo_data['arousal'],
        emotional_intensity=round(emotional_intensity, 3),
        
        age_estimates=demo_data['age_estimates'],
        gender_estimates=demo_data['gender_estimates'],
        motion_intensity=demo_data['motion_intensity'],
        
        nudenet_labels=nn_data['nudenet_labels'],
        nudenet_detection_rate=nn_data['nudenet_detection_rate'],
        has_exposed_content=nn_data['has_exposed_content'],
        
        caption=cap_data['caption'],
        caption_length=cap_data['caption_length'],
        best_still_timestamp=cap_data['best_still_timestamp'],
        vlm_validation_notes=cap_data['vlm_validation_notes'],
        
        priority_score=0.0,  # Will be computed
        training_suitability='low'  # Will be computed
    )
    
    # Compute scores
    record.priority_score = compute_priority_score(record)
    record.training_suitability = get_training_suitability(record.priority_score)
    
    return record


def assemble_all(csv_output: Path, json_output: Path) -> Dict:
    """
    Assemble all scenes into unified dataset.
    
    Returns summary statistics.
    """
    print("Loading all detector outputs...")
    data = load_all_data()
    
    # Report available data
    for key, val in data.items():
        if val:
            count = len(val.get('analyses', val.get('results', [])))
            print(f"  ✓ {key}: {count} entries")
        else:
            print(f"  ✗ {key}: not found")
    
    # Discover scenes
    print("\nDiscovering scenes...")
    scene_names = discover_scenes()
    print(f"  Found {len(scene_names)} scenes")
    
    if not scene_names:
        print("No scenes found!")
        return {'total': 0}
    
    # Assemble records
    print("\nAssembling records...")
    records = []
    for scene_name in scene_names:
        record = assemble_record(scene_name, data)
        records.append(record)
    
    # Sort by priority score (highest first)
    records.sort(key=lambda r: r.priority_score, reverse=True)
    
    # Ensure output directories exist
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    print(f"\nWriting CSV: {csv_output}")
    fieldnames = [f.name for f in UnifiedSceneRecord.__dataclass_fields__.values()]
    
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    
    # Write JSON manifest
    print(f"Writing JSON: {json_output}")
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'total_scenes': len(records),
        'by_suitability': {
            'high': len([r for r in records if r.training_suitability == 'high']),
            'medium': len([r for r in records if r.training_suitability == 'medium']),
            'low': len([r for r in records if r.training_suitability == 'low'])
        },
        'with_captions': len([r for r in records if r.caption]),
        'with_best_still': len([r for r in records if r.best_still_timestamp]),
        'scenes': [asdict(r) for r in records]
    }
    
    with open(json_output, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    summary = {
        'total': len(records),
        'high_priority': manifest['by_suitability']['high'],
        'medium_priority': manifest['by_suitability']['medium'],
        'low_priority': manifest['by_suitability']['low'],
        'with_captions': manifest['with_captions'],
        'with_best_still': manifest['with_best_still']
    }
    
    return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Assemble unified dataset from all detector outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output:
  - unified_dataset.csv: Complete dataset sorted by priority score
  - training_manifest.json: Structured format with metadata

Columns include scene paths, all detector metrics, VLM captions,
[BEST STILL] timestamps, and computed priority scores.

NOTE: This output is FOR USER REVIEW ONLY - contains full labels.
        """
    )
    
    parser.add_argument(
        '--output-csv', '-c',
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"CSV output file (default: {DEFAULT_CSV_OUTPUT})"
    )
    
    parser.add_argument(
        '--output-json', '-j',
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"JSON output file (default: {DEFAULT_JSON_OUTPUT})"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Final Dataset Assembler")
    print("=" * 60)
    print(f"CSV Output:  {args.output_csv}")
    print(f"JSON Output: {args.output_json}")
    print("=" * 60)
    
    summary = assemble_all(args.output_csv, args.output_json)
    
    print("\n" + "=" * 60)
    print("ASSEMBLY COMPLETE")
    print("=" * 60)
    print(f"Total Scenes:    {summary['total']}")
    print(f"High Priority:   {summary['high_priority']}")
    print(f"Medium Priority: {summary['medium_priority']}")
    print(f"Low Priority:    {summary['low_priority']}")
    print(f"With Captions:   {summary['with_captions']}")
    print(f"With Best Still: {summary['with_best_still']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
