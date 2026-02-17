#!/usr/bin/env python3
"""
Data Sanitizer for AI-Safe Statistical Review
==============================================

Creates anonymized statistics from detector outputs, replacing all
content-descriptive labels with generic identifiers (A, B, C...).

This allows the AI assistant to review statistical patterns and
distributions WITHOUT exposure to content-descriptive information.

Output:
- Replaces all text labels with generic IDs (EMO_A, NN_B, CAP_001, etc.)
- Preserves numerical distributions and relationships
- Maintains cluster memberships with anonymized IDs
- Exports sanitized summary statistics

CRITICAL: This is the ONLY output the AI assistant should review.
All other outputs retain full labels for user/VLM consumption.

Usage:
    python data_sanitizer.py [--output FILE]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files (raw detector outputs - labels not exposed to AI)
DETECTIONS_FILE = Path(__file__).parent.parent / "analysis" / "detections.json"
EMOTIONS_FILE = Path(__file__).parent.parent / "analysis" / "emotions.json"
DEMOGRAPHICS_FILE = Path(__file__).parent.parent / "analysis" / "demographics.json"
NUDENET_FILE = Path(__file__).parent.parent / "analysis" / "nudenet.json"
CAPTIONS_FILE = Path(__file__).parent.parent / "analysis" / "captions.json"

# Output file (sanitized - safe for AI review)
DEFAULT_OUTPUT_FILE = Path(__file__).parent.parent / "analysis" / "sanitized_stats.json"

# Mapping file (for user reference only - DO NOT EXPOSE TO AI)
MAPPING_FILE = Path(__file__).parent.parent / "analysis" / "label_mapping.json"


# =============================================================================
# LABEL ANONYMIZATION
# =============================================================================

class LabelAnonymizer:
    """
    Converts content-descriptive labels to generic identifiers.
    
    Mapping scheme:
    - Emotions: EMO_A, EMO_B, EMO_C, ...
    - NudeNet labels: NN_A, NN_B, NN_C, ...
    - Demographics (age): AGE_A, AGE_B, ...
    - Demographics (gender): GEN_A, GEN_B, ...
    - Captions: CAP_001, CAP_002, ...
    - Scenes: SCN_001, SCN_002, ...
    """
    
    def __init__(self):
        self.mappings = {
            'emotion': {},
            'nudenet': {},
            'age': {},
            'gender': {},
            'caption': {},
            'scene': {}
        }
        self.counters = {k: 0 for k in self.mappings}
    
    def _get_next_id(self, category: str, prefix: str) -> str:
        """Generate next anonymous ID for a category."""
        idx = self.counters[category]
        self.counters[category] += 1
        
        if category in ('caption', 'scene'):
            return f"{prefix}_{idx:03d}"
        else:
            # Use A-Z, then AA, AB, etc.
            if idx < 26:
                suffix = chr(ord('A') + idx)
            else:
                first = chr(ord('A') + (idx // 26) - 1)
                second = chr(ord('A') + (idx % 26))
                suffix = f"{first}{second}"
            return f"{prefix}_{suffix}"
    
    def anonymize(self, category: str, label: str, prefix: str) -> str:
        """
        Get or create anonymous ID for a label.
        
        Args:
            category: Label category (emotion, nudenet, etc.)
            label: Original label text
            prefix: Prefix for anonymous ID
            
        Returns:
            Anonymous identifier
        """
        if label not in self.mappings[category]:
            self.mappings[category][label] = self._get_next_id(category, prefix)
        return self.mappings[category][label]
    
    def get_reverse_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get reverse mapping (anonymous -> original).
        
        FOR USER REFERENCE ONLY - DO NOT EXPOSE TO AI
        """
        reverse = {}
        for category, mapping in self.mappings.items():
            reverse[category] = {v: k for k, v in mapping.items()}
        return reverse


# =============================================================================
# DATA EXTRACTION AND SANITIZATION
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


def sanitize_detections(data: Dict, anonymizer: LabelAnonymizer) -> Dict:
    """
    Extract sanitized statistics from YOLO detections.
    
    Preserves: coverage %, confidence scores, person counts
    Anonymizes: scene paths
    """
    if not data or 'analyses' not in data:
        return {'available': False}
    
    stats = {
        'available': True,
        'total_scenes': len(data['analyses']),
        'coverage_distribution': [],  # List of coverage percentages
        'confidence_distribution': [],  # List of confidence scores
        'person_count_distribution': [],  # List of max person counts
        'scenes_with_persons': 0,
        'scenes_without_persons': 0
    }
    
    for analysis in data['analyses']:
        coverage = analysis.get('detection_coverage', 0)
        confidence = analysis.get('avg_confidence', 0)
        max_persons = analysis.get('max_persons_detected', 0)
        
        stats['coverage_distribution'].append(round(coverage, 3))
        stats['confidence_distribution'].append(round(confidence, 3))
        stats['person_count_distribution'].append(max_persons)
        
        if analysis.get('person_present', False):
            stats['scenes_with_persons'] += 1
        else:
            stats['scenes_without_persons'] += 1
    
    return stats


def sanitize_emotions(data: Dict, anonymizer: LabelAnonymizer) -> Dict:
    """
    Extract sanitized statistics from emotion analysis.
    
    Preserves: valence/arousal scores, emotion frequencies (anonymized)
    Anonymizes: emotion labels, scene paths
    """
    if not data or 'analyses' not in data:
        return {'available': False}
    
    stats = {
        'available': True,
        'total_scenes': len(data['analyses']),
        'valence_distribution': [],
        'arousal_distribution': [],
        'emotion_frequencies': Counter(),  # Anonymized emotion -> count
        'scenes_with_faces': 0
    }
    
    for analysis in data['analyses']:
        valence = analysis.get('mean_valence', 0)
        arousal = analysis.get('mean_arousal', 0)
        dominant = analysis.get('dominant_emotion', 'unknown')
        
        stats['valence_distribution'].append(round(valence, 3))
        stats['arousal_distribution'].append(round(arousal, 3))
        
        # Anonymize emotion label
        anon_emotion = anonymizer.anonymize('emotion', dominant, 'EMO')
        stats['emotion_frequencies'][anon_emotion] += 1
        
        if analysis.get('frames_with_faces', 0) > 0:
            stats['scenes_with_faces'] += 1
    
    # Convert Counter to dict for JSON
    stats['emotion_frequencies'] = dict(stats['emotion_frequencies'])
    
    return stats


def sanitize_demographics(data: Dict, anonymizer: LabelAnonymizer) -> Dict:
    """
    Extract sanitized statistics from demographics analysis.
    
    Preserves: age/gender distributions (anonymized), motion scores
    Anonymizes: age categories, gender labels, scene paths
    """
    if not data or 'analyses' not in data:
        return {'available': False}
    
    stats = {
        'available': True,
        'total_scenes': len(data['analyses']),
        'motion_distribution': [],
        'age_frequencies': Counter(),  # Anonymized age -> count
        'gender_frequencies': Counter(),  # Anonymized gender -> count
    }
    
    for analysis in data['analyses']:
        motion = analysis.get('motion_intensity', 0)
        stats['motion_distribution'].append(round(motion, 3))
        
        # Anonymize age categories
        for age_cat in analysis.get('age_categories', []):
            anon_age = anonymizer.anonymize('age', age_cat, 'AGE')
            stats['age_frequencies'][anon_age] += 1
        
        # Anonymize genders
        for gender in analysis.get('genders', []):
            anon_gender = anonymizer.anonymize('gender', gender, 'GEN')
            stats['gender_frequencies'][anon_gender] += 1
    
    stats['age_frequencies'] = dict(stats['age_frequencies'])
    stats['gender_frequencies'] = dict(stats['gender_frequencies'])
    
    return stats


def sanitize_nudenet(data: Dict, anonymizer: LabelAnonymizer) -> Dict:
    """
    Extract sanitized statistics from NudeNet analysis.
    
    Preserves: detection rates, label frequencies (anonymized)
    Anonymizes: body part labels, scene paths
    """
    if not data or 'analyses' not in data:
        return {'available': False}
    
    stats = {
        'available': True,
        'total_scenes': len(data['analyses']),
        'detection_rate_distribution': [],
        'label_frequencies': Counter(),  # Anonymized label -> total count
        'scenes_with_detections': 0
    }
    
    for scene_name, scene_data in data['analyses'].items():
        det_rate = scene_data.get('detection_rate', 0)
        stats['detection_rate_distribution'].append(round(det_rate, 3))
        
        if det_rate > 0:
            stats['scenes_with_detections'] += 1
        
        # Anonymize labels
        for label, count in scene_data.get('label_counts', {}).items():
            anon_label = anonymizer.anonymize('nudenet', label, 'NN')
            stats['label_frequencies'][anon_label] += count
    
    stats['label_frequencies'] = dict(stats['label_frequencies'])
    
    return stats


def sanitize_captions(data: Dict, anonymizer: LabelAnonymizer) -> Dict:
    """
    Extract sanitized statistics from VLM captions.
    
    Preserves: caption lengths, success rates
    Anonymizes: caption content (replaced with length only), scene paths
    
    NOTE: Caption TEXT is completely removed - only metadata preserved
    """
    if not data or 'results' not in data:
        return {'available': False}
    
    stats = {
        'available': True,
        'total_clips': data.get('total_clips', 0),
        'successful': data.get('successful', 0),
        'failed': data.get('failed', 0),
        'caption_length_distribution': [],  # Length in characters
        'word_count_distribution': []  # Approximate word counts
    }
    
    for result in data['results']:
        caption = result.get('caption')
        if caption:
            stats['caption_length_distribution'].append(len(caption))
            stats['word_count_distribution'].append(len(caption.split()))
    
    return stats


def compute_cross_correlations(
    detections: Dict,
    emotions: Dict,
    demographics: Dict,
    nudenet: Dict
) -> Dict:
    """
    Compute statistical correlations between detector outputs.
    
    All correlations are numerical only - no content labels.
    """
    correlations = {
        'coverage_vs_valence': None,
        'motion_vs_arousal': None,
        'detection_rate_vs_coverage': None
    }
    
    # Only compute if we have enough data
    try:
        import numpy as np
        
        # Coverage vs Valence
        if (detections.get('available') and emotions.get('available') and
            len(detections['coverage_distribution']) == len(emotions['valence_distribution'])):
            cov = np.array(detections['coverage_distribution'])
            val = np.array(emotions['valence_distribution'])
            if len(cov) > 2:
                corr = np.corrcoef(cov, val)[0, 1]
                correlations['coverage_vs_valence'] = round(float(corr), 3) if not np.isnan(corr) else None
        
        # Motion vs Arousal
        if (demographics.get('available') and emotions.get('available') and
            len(demographics['motion_distribution']) == len(emotions['arousal_distribution'])):
            mot = np.array(demographics['motion_distribution'])
            aro = np.array(emotions['arousal_distribution'])
            if len(mot) > 2:
                corr = np.corrcoef(mot, aro)[0, 1]
                correlations['motion_vs_arousal'] = round(float(corr), 3) if not np.isnan(corr) else None
        
        # NudeNet detection rate vs YOLO coverage
        if (nudenet.get('available') and detections.get('available') and
            len(nudenet['detection_rate_distribution']) == len(detections['coverage_distribution'])):
            nn_rate = np.array(nudenet['detection_rate_distribution'])
            yolo_cov = np.array(detections['coverage_distribution'])
            if len(nn_rate) > 2:
                corr = np.corrcoef(nn_rate, yolo_cov)[0, 1]
                correlations['detection_rate_vs_coverage'] = round(float(corr), 3) if not np.isnan(corr) else None
                
    except ImportError:
        print("Warning: numpy not available, skipping correlations")
    except Exception as e:
        print(f"Warning: Could not compute correlations: {e}")
    
    return correlations


def compute_distribution_stats(values: List[float]) -> Dict:
    """Compute basic distribution statistics for a list of values."""
    if not values:
        return {'count': 0}
    
    try:
        import numpy as np
        arr = np.array(values)
        return {
            'count': len(arr),
            'mean': round(float(np.mean(arr)), 3),
            'std': round(float(np.std(arr)), 3),
            'min': round(float(np.min(arr)), 3),
            'max': round(float(np.max(arr)), 3),
            'median': round(float(np.median(arr)), 3),
            'q25': round(float(np.percentile(arr, 25)), 3),
            'q75': round(float(np.percentile(arr, 75)), 3)
        }
    except ImportError:
        # Fallback without numpy
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            'count': n,
            'mean': round(sum(values) / n, 3),
            'min': round(min(values), 3),
            'max': round(max(values), 3),
            'median': round(sorted_vals[n // 2], 3)
        }


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def generate_sanitized_stats(output_file: Path) -> Dict:
    """
    Generate fully sanitized statistics file.
    
    This is the ONLY output safe for AI assistant review.
    """
    print("Loading detector outputs...")
    
    # Initialize anonymizer
    anonymizer = LabelAnonymizer()
    
    # Load and sanitize each data source
    print("  Processing detections...")
    detections_raw = load_json_safe(DETECTIONS_FILE)
    detections = sanitize_detections(detections_raw, anonymizer)
    
    print("  Processing emotions...")
    emotions_raw = load_json_safe(EMOTIONS_FILE)
    emotions = sanitize_emotions(emotions_raw, anonymizer)
    
    print("  Processing demographics...")
    demographics_raw = load_json_safe(DEMOGRAPHICS_FILE)
    demographics = sanitize_demographics(demographics_raw, anonymizer)
    
    print("  Processing nudenet...")
    nudenet_raw = load_json_safe(NUDENET_FILE)
    nudenet = sanitize_nudenet(nudenet_raw, anonymizer)
    
    print("  Processing captions...")
    captions_raw = load_json_safe(CAPTIONS_FILE)
    captions = sanitize_captions(captions_raw, anonymizer)
    
    # Compute distribution summaries
    print("\nComputing distribution statistics...")
    distribution_summaries = {
        'coverage': compute_distribution_stats(detections.get('coverage_distribution', [])),
        'confidence': compute_distribution_stats(detections.get('confidence_distribution', [])),
        'valence': compute_distribution_stats(emotions.get('valence_distribution', [])),
        'arousal': compute_distribution_stats(emotions.get('arousal_distribution', [])),
        'motion': compute_distribution_stats(demographics.get('motion_distribution', [])),
        'nudenet_detection_rate': compute_distribution_stats(nudenet.get('detection_rate_distribution', [])),
        'caption_length': compute_distribution_stats(captions.get('caption_length_distribution', []))
    }
    
    # Compute cross-correlations
    print("Computing correlations...")
    correlations = compute_cross_correlations(detections, emotions, demographics, nudenet)
    
    # Build final sanitized output
    sanitized = {
        '_meta': {
            'description': 'Sanitized statistics - safe for AI review',
            'note': 'All content labels replaced with generic identifiers',
            'label_categories': {
                'EMO_*': 'Emotion categories',
                'NN_*': 'NudeNet detection labels',
                'AGE_*': 'Age categories',
                'GEN_*': 'Gender categories'
            }
        },
        'detections': detections,
        'emotions': emotions,
        'demographics': demographics,
        'nudenet': nudenet,
        'captions': captions,
        'distribution_summaries': distribution_summaries,
        'correlations': correlations
    }
    
    # Save sanitized stats
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(sanitized, f, indent=2)
    
    print(f"\nSanitized stats saved to: {output_file}")
    
    # Save label mapping (FOR USER ONLY - not for AI)
    mapping = anonymizer.get_reverse_mapping()
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Label mapping saved to: {MAPPING_FILE}")
    print("  (Label mapping is FOR USER REFERENCE ONLY)")
    
    return sanitized


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate sanitized statistics for AI-safe review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates an anonymized summary that the AI assistant can review
without exposure to content-descriptive labels.

Output:
  - sanitized_stats.json: Safe for AI review
  - label_mapping.json: FOR USER ONLY (maps anonymous IDs to real labels)

The AI assistant should ONLY read sanitized_stats.json.
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Sanitizer for AI-Safe Review")
    print("=" * 60)
    print(f"Output: {args.output}")
    print("=" * 60)
    
    sanitized = generate_sanitized_stats(args.output)
    
    # Print summary (safe for AI to see)
    print("\n" + "=" * 60)
    print("SANITIZED SUMMARY")
    print("=" * 60)
    
    if sanitized['detections'].get('available'):
        print(f"Detections: {sanitized['detections']['total_scenes']} scenes")
        print(f"  With persons: {sanitized['detections']['scenes_with_persons']}")
    
    if sanitized['emotions'].get('available'):
        print(f"Emotions: {sanitized['emotions']['total_scenes']} scenes")
        print(f"  Unique categories: {len(sanitized['emotions']['emotion_frequencies'])}")
    
    if sanitized['demographics'].get('available'):
        print(f"Demographics: {sanitized['demographics']['total_scenes']} scenes")
    
    if sanitized['nudenet'].get('available'):
        print(f"NudeNet: {sanitized['nudenet']['total_scenes']} scenes")
        print(f"  With detections: {sanitized['nudenet']['scenes_with_detections']}")
    
    if sanitized['captions'].get('available'):
        print(f"Captions: {sanitized['captions']['total_clips']} clips")
        print(f"  Successful: {sanitized['captions']['successful']}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
