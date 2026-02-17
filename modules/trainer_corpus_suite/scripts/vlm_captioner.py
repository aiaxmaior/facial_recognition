#!/usr/bin/env python3
"""
VLM Video Captioner with Detector Context
==========================================

Sends video clips to vLLM for captioning, prepending all detector outputs
as context for each clip. The VLM validates/rejects detector data as part
of its analysis.

Prerequisites:
- vLLM server running with Qwen3-VL model
- Detector outputs in analysis/ directory:
  - detections.json (YOLO person detection)
  - emotions.json (DeepFace emotion analysis)
  - demographics.json (age/gender/motion analysis)
  - nudenet.json (body part detection)

CRITICAL: This script does NOT interpret video content directly.
All context is provided by automated detectors, VLM does the visual analysis.

Usage:
    python vlm_captioner.py [--vlm-dir DIR] [--output FILE] [--api-url URL]
"""

import json
import sys
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

# vLLM API settings
DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL_NAME = "QDrive_AI"

# Input/Output paths
DEFAULT_VLM_DIR = Path(__file__).parent.parent / "vlm_copies"
DEFAULT_OUTPUT_FILE = Path(__file__).parent.parent / "analysis" / "captions.json"
DEFAULT_PROMPT_FILE = Path(__file__).parent.parent / "prompt.txt"

# Detector output files
DETECTIONS_FILE = Path(__file__).parent.parent / "analysis" / "detections.json"
EMOTIONS_FILE = Path(__file__).parent.parent / "analysis" / "emotions.json"
DEMOGRAPHICS_FILE = Path(__file__).parent.parent / "analysis" / "demographics.json"
NUDENET_FILE = Path(__file__).parent.parent / "analysis" / "nudenet.json"
# HAR (Human Action Recognition) output files
HAR_DETECTIONS_FILE = Path(__file__).parent.parent / "analysis" / "har_detections.json"
HAR_ACTIONS_FILE = Path(__file__).parent.parent / "analysis" / "har_actions.json"

# Processing settings
BATCH_SIZE = 1  # Process one video at a time for video models
REQUEST_TIMEOUT = 300  # 5 minutes per video
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Video extensions
VIDEO_EXTENSIONS = {'.mp4', '.MP4'}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PersonDetail:
    """Per-person demographic detail."""
    person_idx: int
    mean_age: Optional[float]
    age_range: Optional[Tuple[float, float]]
    dominant_gender: Optional[str]
    dominant_emotion: Optional[str]
    appearances: int
    total_frames: int


@dataclass
class ActionPrediction:
    """HAR action prediction with rank."""
    label: str
    score: float
    rank: int


@dataclass
class SceneContext:
    """Aggregated detector context for a single scene."""
    scene_name: str
    vlm_video_path: str
    
    # YOLO detection
    person_coverage: float
    person_confidence: float
    max_persons: int
    
    # Emotion analysis
    dominant_emotion: str
    emotion_confidence: float
    valence: float
    arousal: float
    
    # Demographics
    age_estimates: List[str]
    gender_estimates: List[str]
    motion_intensity: float
    
    # NudeNet
    nudenet_labels: Dict[str, int]
    nudenet_detection_rate: float
    
    # Per-person details
    person_details: List[PersonDetail] = None
    
    # HAR (Human Action Recognition) - NEW
    dominant_actions: List[ActionPrediction] = None


# =============================================================================
# CONTEXT LOADING
# =============================================================================

def load_json_safe(file_path: Path) -> Optional[Dict]:
    """Load JSON file if it exists, return None otherwise."""
    if file_path.exists():
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    return None


def load_all_detector_outputs() -> Dict[str, Dict]:
    """
    Load all detector outputs into a unified structure.
    
    Returns:
        Dict with keys: detections, emotions, demographics, nudenet, har
    """
    return {
        'detections': load_json_safe(DETECTIONS_FILE),
        'emotions': load_json_safe(EMOTIONS_FILE),
        'demographics': load_json_safe(DEMOGRAPHICS_FILE),
        'nudenet': load_json_safe(NUDENET_FILE),
        'har_detections': load_json_safe(HAR_DETECTIONS_FILE),
        'har_actions': load_json_safe(HAR_ACTIONS_FILE),
    }


def get_scene_context(scene_name: str, vlm_path: str, detector_data: Dict) -> SceneContext:
    """
    Build context for a specific scene from all detector outputs.
    
    Args:
        scene_name: Base name of the scene (without _vlm suffix)
        vlm_path: Path to the VLM video file
        detector_data: Dict of all detector outputs
        
    Returns:
        SceneContext with aggregated data
    """
    # Default values
    context = SceneContext(
        scene_name=scene_name,
        vlm_video_path=vlm_path,
        person_coverage=0.0,
        person_confidence=0.0,
        max_persons=0,
        dominant_emotion="unknown",
        emotion_confidence=0.0,
        valence=0.0,
        arousal=0.0,
        age_estimates=[],
        gender_estimates=[],
        motion_intensity=0.0,
        nudenet_labels={},
        nudenet_detection_rate=0.0
    )
    
    # Extract YOLO detection data
    detections = detector_data.get('detections')
    if detections and 'analyses' in detections:
        for analysis in detections['analyses']:
            # Match by scene path (may need fuzzy matching)
            if scene_name in analysis.get('scene_path', ''):
                context.person_coverage = analysis.get('detection_coverage', 0.0)
                context.person_confidence = analysis.get('avg_confidence', 0.0)
                context.max_persons = analysis.get('max_persons_detected', 0)
                break
    
    # Extract emotion data
    emotions = detector_data.get('emotions')
    if emotions and 'analyses' in emotions:
        for analysis in emotions['analyses']:
            if scene_name in analysis.get('scene_path', ''):
                context.dominant_emotion = analysis.get('dominant_emotion', 'unknown')
                context.emotion_confidence = analysis.get('emotion_confidence', 0.0)
                context.valence = analysis.get('mean_valence', 0.0)
                context.arousal = analysis.get('mean_arousal', 0.0)
                break
    
    # Extract demographics data
    demographics = detector_data.get('demographics')
    if demographics and 'analyses' in demographics:
        for analysis in demographics['analyses']:
            if scene_name in analysis.get('scene_path', ''):
                context.age_estimates = analysis.get('age_categories', [])
                context.gender_estimates = analysis.get('genders', [])
                context.motion_intensity = analysis.get('motion_intensity', 0.0)
                
                # Extract per-person details from frame_demographics
                frame_demos = analysis.get('frame_demographics', [])
                if frame_demos:
                    person_data = _extract_person_details(frame_demos)
                    context.person_details = person_data
                break
    
    # Extract NudeNet data
    nudenet = detector_data.get('nudenet')
    if nudenet and 'analyses' in nudenet:
        scene_data = nudenet['analyses'].get(scene_name, {})
        context.nudenet_labels = scene_data.get('label_counts', {})
        context.nudenet_detection_rate = scene_data.get('detection_rate', 0.0)
    
    # Extract HAR (Human Action Recognition) data
    har_detections = detector_data.get('har_detections')
    har_actions = detector_data.get('har_actions')
    
    # Try har_detections first (already processed format)
    if har_detections and 'analyses' in har_detections:
        for analysis in har_detections['analyses']:
            if scene_name in analysis.get('scene_path', ''):
                dominant_actions = analysis.get('dominant_actions', [])
                if dominant_actions:
                    context.dominant_actions = [
                        ActionPrediction(
                            label=a['label'],
                            score=a['score'],
                            rank=a['rank']
                        )
                        for a in dominant_actions[:3]
                    ]
                break
    
    # Fallback to har_actions.json clips array
    elif har_actions and 'clips' in har_actions:
        for clip in har_actions['clips']:
            if scene_name in clip.get('clip_path', ''):
                top_k = clip.get('predictions', {}).get('top_k', [])
                if top_k:
                    context.dominant_actions = [
                        ActionPrediction(
                            label=p['label'],
                            score=p['score'],
                            rank=i + 1
                        )
                        for i, p in enumerate(top_k[:3])
                    ]
                break
    
    return context


def _extract_person_details(frame_demographics: List[Dict]) -> List[PersonDetail]:
    """
    Extract per-person summary data from frame demographics.
    
    Aggregates person data across frames to build per-person profiles.
    
    Args:
        frame_demographics: List of frame data with persons array
        
    Returns:
        List of PersonDetail for each unique person slot
    """
    # Aggregate data by person_idx
    person_data = {}  # {person_idx: {ages: [], genders: [], emotions: [], appearances: int}}
    total_frames = len(frame_demographics)
    
    for frame in frame_demographics:
        persons = frame.get('persons', [])
        for idx, person in enumerate(persons):
            if idx not in person_data:
                person_data[idx] = {
                    'ages': [],
                    'genders': [],
                    'emotions': [],
                    'appearances': 0
                }
            
            person_data[idx]['appearances'] += 1
            
            age = person.get('age_combined')
            if age is not None:
                person_data[idx]['ages'].append(age)
            
            gender = person.get('gender_combined')
            if gender:
                person_data[idx]['genders'].append(gender)
            
            # Note: per-face emotion would come from emotions.json faces array
            # For now, we don't have direct face-to-person mapping here
    
    # Build PersonDetail objects
    details = []
    for person_idx, data in sorted(person_data.items()):
        ages = data['ages']
        genders = data['genders']
        
        mean_age = sum(ages) / len(ages) if ages else None
        age_range = (min(ages), max(ages)) if ages else None
        
        # Find dominant gender
        dominant_gender = None
        if genders:
            gender_counts = {}
            for g in genders:
                gender_counts[g] = gender_counts.get(g, 0) + 1
            dominant_gender = max(gender_counts, key=gender_counts.get)
        
        details.append(PersonDetail(
            person_idx=person_idx,
            mean_age=mean_age,
            age_range=age_range,
            dominant_gender=dominant_gender,
            dominant_emotion=None,  # Not available without face-person linking
            appearances=data['appearances'],
            total_frames=total_frames
        ))
    
    return details


def format_context_prompt(context: SceneContext) -> str:
    """
    Format detector context as prepended prompt text.
    
    This text is prepended to the user's prompt for each clip.
    Includes caveat about potential detector errors.
    
    Args:
        context: SceneContext with detector data
        
    Returns:
        Formatted context string
    """
    lines = [
        "=" * 50,
        "AUTOMATED DETECTOR CONTEXT",
        "=" * 50,
        f"Scene: {context.scene_name}",
        "",
        "YOLO Person Detection:",
        f"  - Coverage: {context.person_coverage*100:.1f}% of frames",
        f"  - Confidence: {context.person_confidence:.2f}",
        f"  - Max persons detected: {context.max_persons}",
        "",
        "Emotion Analysis (DeepFace):",
        f"  - Dominant: {context.dominant_emotion} ({context.emotion_confidence:.2f})",
        f"  - Valence: {context.valence:.2f} (-1=aversion, +1=excitement)",
        f"  - Arousal: {context.arousal:.2f} (0=calm, 1=intense)",
        "",
    ]
    
    # Per-Person Demographics (NEW - richer context for VLM)
    if context.person_details and len(context.person_details) > 0:
        lines.append("Demographics (Per-Person):")
        lines.append(f"  Total persons tracked: {len(context.person_details)}")
        lines.append("")
        
        for person in context.person_details:
            age_str = f"~{person.mean_age:.0f} years old" if person.mean_age else "age unknown"
            gender_str = person.dominant_gender or "gender unknown"
            frame_str = f"{person.appearances}/{person.total_frames} frames"
            
            lines.append(f"  Person {person.person_idx + 1}: {age_str}, {gender_str}")
            lines.append(f"    - Appears in {frame_str} ({person.appearances/person.total_frames*100:.0f}% coverage)")
            
            if person.age_range and person.age_range[0] != person.age_range[1]:
                lines.append(f"    - Age range: {person.age_range[0]:.0f}-{person.age_range[1]:.0f}")
        
        lines.append("")
    else:
        # Fallback to aggregated demographics
        lines.extend([
            "Demographics (Aggregated):",
            f"  - Age estimates: {', '.join(context.age_estimates) if context.age_estimates else 'N/A'}",
            f"  - Gender estimates: {', '.join(context.gender_estimates) if context.gender_estimates else 'N/A'}",
            "",
        ])
    
    lines.append(f"Motion intensity: {context.motion_intensity:.2f}")
    lines.append("")
    lines.append("NudeNet Detections:")
    
    if context.nudenet_labels:
        for label, count in sorted(context.nudenet_labels.items(), key=lambda x: -x[1]):
            lines.append(f"  - {label}: {count} frames")
        lines.append(f"  - Detection rate: {context.nudenet_detection_rate*100:.1f}%")
    else:
        lines.append("  - No detections")
    
    # HAR (Human Action Recognition) - Top-3 ranked actions
    lines.append("")
    lines.append("Human Action Recognition (HAR):")
    
    if context.dominant_actions and len(context.dominant_actions) > 0:
        for action in context.dominant_actions:
            confidence_pct = action.score * 100
            lines.append(f"  {action.rank}. {action.label}: {confidence_pct:.1f}% confidence")
    else:
        lines.append("  - No action predictions available")
    
    lines.extend([
        "",
        "=" * 50,
        "NOTE: The above detector outputs are automated and may",
        "contain errors. Part of your task is to VALIDATE or REJECT",
        "this data based on your visual analysis of the clip.",
        "=" * 50,
        "",
    ])
    
    return "\n".join(lines)


# =============================================================================
# VLM API INTERACTION
# =============================================================================

def load_prompt_file(prompt_path: Path) -> str:
    """
    Load the user's prompt file.
    
    Note: The content of this file is passed directly to the VLM
    without being read/interpreted by this script's author.
    
    Args:
        prompt_path: Path to prompt.txt
        
    Returns:
        Prompt text content
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def encode_video_for_api(video_path: Path) -> str:
    """
    Encode video file for vLLM API (base64 or file path depending on API).
    
    For local vLLM with --allowed-local-media-path, we can use file:// URLs.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Encoded video reference for API
    """
    # Use file:// URL for local vLLM with allowed media path
    return f"file://{video_path.absolute()}"


def call_vlm_api(
    video_path: Path,
    context_prompt: str,
    user_prompt: str,
    api_url: str,
    model_name: str
) -> Optional[str]:
    """
    Call vLLM API with video and prompts.
    
    Args:
        video_path: Path to video file
        context_prompt: Prepended detector context
        user_prompt: User's analysis prompt
        api_url: vLLM API endpoint
        model_name: Model name configured in vLLM
        
    Returns:
        VLM response text or None on failure
    """
    video_url = encode_video_for_api(video_path)
    
    # Combine context + user prompt
    full_prompt = f"{context_prompt}\n{user_prompt}"
    
    # Build API request
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url}
                    },
                    {
                        "type": "text",
                        "text": full_prompt
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            print(f"  API error: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
        except (KeyError, IndexError) as e:
            print(f"  Response parsing error: {e}")
            return None
    
    return None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_all_clips(
    vlm_dir: Path,
    output_file: Path,
    prompt_file: Path,
    api_url: str,
    model_name: str
) -> Dict:
    """
    Process all VLM clips with detector context.
    
    Args:
        vlm_dir: Directory containing low-FPS VLM copies
        output_file: Path for output JSON
        prompt_file: Path to user's prompt file
        api_url: vLLM API endpoint
        model_name: Model name for API
        
    Returns:
        Processing summary
    """
    # Load all detector outputs
    print("Loading detector outputs...")
    detector_data = load_all_detector_outputs()
    
    # Report what's available
    for key, data in detector_data.items():
        if data:
            count = len(data.get('analyses', data.get('analyses', [])))
            print(f"  ✓ {key}: {count} entries")
        else:
            print(f"  ✗ {key}: not found")
    
    # Load user prompt (we pass it through without reading)
    print(f"\nLoading prompt from: {prompt_file}")
    user_prompt = load_prompt_file(prompt_file)
    print(f"  Prompt loaded ({len(user_prompt)} characters)")
    
    # Find all VLM clips
    vlm_clips = sorted([
        f for f in vlm_dir.iterdir()
        if f.suffix.lower() in {e.lower() for e in VIDEO_EXTENSIONS}
    ])
    print(f"\nFound {len(vlm_clips)} VLM clips to process")
    
    if not vlm_clips:
        return {'total_clips': 0, 'successful': 0, 'failed': 0}
    
    # Process each clip
    results = []
    successful = 0
    failed = 0
    
    for clip_path in tqdm(vlm_clips, desc="Processing clips"):
        # Extract scene name (remove _vlm suffix)
        scene_name = clip_path.stem.replace('_vlm', '')
        
        # Build context from detector outputs
        context = get_scene_context(scene_name, str(clip_path), detector_data)
        context_prompt = format_context_prompt(context)
        
        # Call VLM API
        response = call_vlm_api(
            clip_path,
            context_prompt,
            user_prompt,
            api_url,
            model_name
        )
        
        if response:
            results.append({
                'scene_name': scene_name,
                'vlm_path': str(clip_path),
                'context': asdict(context),
                'caption': response
            })
            successful += 1
        else:
            results.append({
                'scene_name': scene_name,
                'vlm_path': str(clip_path),
                'context': asdict(context),
                'caption': None,
                'error': 'API call failed'
            })
            failed += 1
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'total_clips': len(vlm_clips),
            'successful': successful,
            'failed': failed,
            'model': model_name,
            'api_url': api_url,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return {
        'total_clips': len(vlm_clips),
        'successful': successful,
        'failed': failed
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VLM Video Captioner with Detector Context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  1. Start vLLM server:
     vllm serve /path/to/model --served-model-name QDrive_AI --port 8000 \\
       --allowed-local-media-path /
  
  2. Run all detectors first:
     python emotion_detector.py
     python demographics_detector.py
     python nudenet_batch_processor.py

Examples:
  # Process with defaults
  python vlm_captioner.py
  
  # Custom API endpoint
  python vlm_captioner.py --api-url http://localhost:8080/v1/chat/completions
  
  # Custom prompt file
  python vlm_captioner.py --prompt-file custom_prompt.txt
        """
    )
    
    parser.add_argument(
        '--vlm-dir', '-v',
        type=Path,
        default=DEFAULT_VLM_DIR,
        help=f"Directory with VLM video copies (default: {DEFAULT_VLM_DIR})"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})"
    )
    
    parser.add_argument(
        '--prompt-file', '-p',
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"User prompt file (default: {DEFAULT_PROMPT_FILE})"
    )
    
    parser.add_argument(
        '--api-url', '-a',
        type=str,
        default=DEFAULT_API_URL,
        help=f"vLLM API endpoint (default: {DEFAULT_API_URL})"
    )
    
    parser.add_argument(
        '--model-name', '-m',
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name for API (default: {DEFAULT_MODEL_NAME})"
    )
    
    args = parser.parse_args()
    
    # Print config
    print("=" * 60)
    print("VLM Video Captioner with Detector Context")
    print("=" * 60)
    print(f"VLM Directory: {args.vlm_dir}")
    print(f"Output File:   {args.output}")
    print(f"Prompt File:   {args.prompt_file}")
    print(f"API URL:       {args.api_url}")
    print(f"Model Name:    {args.model_name}")
    print("=" * 60)
    
    # Validate inputs
    if not args.vlm_dir.exists():
        print(f"Error: VLM directory does not exist: {args.vlm_dir}")
        sys.exit(1)
    
    if not args.prompt_file.exists():
        print(f"Error: Prompt file does not exist: {args.prompt_file}")
        sys.exit(1)
    
    # Process
    summary = process_all_clips(
        args.vlm_dir,
        args.output,
        args.prompt_file,
        args.api_url,
        args.model_name
    )
    
    # Report
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Clips:  {summary['total_clips']}")
    print(f"Successful:   {summary['successful']}")
    print(f"Failed:       {summary['failed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
