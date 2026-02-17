#!/usr/bin/env python3
"""
Detection Configuration - Abstract Label References
====================================================

This module centralizes all classification label references to enable:
1. Easy swapping of label sets for different domains
2. Clean professional codebase without explicit content labels
3. Consistent label handling across all detection scripts

Labels are loaded from external annotation files at runtime.
This file contains NO explicit content labels.

Usage:
    from detection_config import get_har_labels, get_nudenet_labels, HAR_CONFIG, NUDENET_CONFIG
"""

from pathlib import Path
from typing import List, Dict, Optional
import json

# =============================================================================
# CONFIGURATION PATHS
# =============================================================================

# Base paths (relative to this file)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_ANALYSIS_DIR = _PROJECT_DIR / "analysis"

# HAR Configuration
HAR_CONFIG = {
    'enabled': True,
    'model_name': 'HAR',  # Human Action Recognition
    'description': 'Video action classification model',
    'annotation_dir': None,  # Set at runtime if available
    'modalities': ['rgb', 'pose', 'audio'],
    'modality_weights': [0.5, 0.6, 1.0],
    'top_k': 5,
}

# NudeNet Configuration  
NUDENET_CONFIG = {
    'enabled': True,
    'model_name': 'BodyDetector',
    'description': 'Body region detection model',
    'model_size': 640,
    'min_confidence': 0.3,
    # Label categories loaded from external source
    'label_categories': {
        'primary': [],      # Primary detection labels
        'secondary': [],    # Secondary detection labels
    }
}

# =============================================================================
# LABEL LOADING FUNCTIONS
# =============================================================================

def load_labels_from_file(filepath: Path) -> List[str]:
    """
    Load labels from an annotation file (one label per line).
    
    Args:
        filepath: Path to annotation file
        
    Returns:
        List of label strings
    """
    if not filepath.exists():
        return []
    
    with open(filepath) as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_labels_from_json(filepath: Path, key: str = 'labels') -> List[str]:
    """
    Load labels from a JSON configuration file.
    
    Args:
        filepath: Path to JSON file
        key: Key containing label list
        
    Returns:
        List of label strings
    """
    if not filepath.exists():
        return []
    
    with open(filepath) as f:
        data = json.load(f)
    
    return data.get(key, [])


def get_har_labels(annotation_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Get HAR labels from annotation files.
    
    Args:
        annotation_dir: Directory containing annotation files
        
    Returns:
        Dict with 'rgb', 'pose', 'audio' label lists
    """
    if annotation_dir is None:
        return {'rgb': [], 'pose': [], 'audio': []}
    
    return {
        'rgb': load_labels_from_file(annotation_dir / "annotations.txt"),
        'pose': load_labels_from_file(annotation_dir / "annotations_pose.txt"),
        'audio': load_labels_from_file(annotation_dir / "annotations_audio.txt"),
    }


def get_nudenet_labels(config_file: Optional[Path] = None) -> List[str]:
    """
    Get NudeNet detection labels from configuration.
    
    If no config file provided, returns empty list.
    Labels should be defined in external configuration, not hardcoded.
    
    Args:
        config_file: Optional path to label configuration JSON
        
    Returns:
        List of detection label strings
    """
    if config_file and config_file.exists():
        return load_labels_from_json(config_file, 'detection_labels')
    
    # Return empty - labels must be configured externally
    return []


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_label_category(label: str, prefix: str = 'LABEL') -> str:
    """
    Generate anonymous label reference.
    
    Args:
        label: Original label string
        prefix: Prefix for anonymous label
        
    Returns:
        Anonymous label like "LABEL_001"
    """
    # Hash-based anonymization for consistency
    label_hash = abs(hash(label)) % 1000
    return f"{prefix}_{label_hash:03d}"


def is_primary_detection(label: str, primary_indicators: List[str] = None) -> bool:
    """
    Check if a label represents a primary detection category.
    
    Args:
        label: Label to check
        primary_indicators: List of substrings indicating primary category
        
    Returns:
        True if label is primary category
    """
    if primary_indicators is None:
        primary_indicators = []  # No hardcoded indicators
    
    return any(ind in label for ind in primary_indicators)


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config() -> Dict[str, bool]:
    """
    Validate that required configuration is available.
    
    Returns:
        Dict of component availability
    """
    return {
        'har_available': HAR_CONFIG['enabled'],
        'nudenet_available': NUDENET_CONFIG['enabled'],
        'analysis_dir_exists': _ANALYSIS_DIR.exists(),
    }
