#!/usr/bin/env python3
"""
Audio Analysis from Original Videos
====================================

Extracts audio segments from original videos using scene timecodes,
then analyzes each segment to produce per-scene audio data.

This is useful when scene clips don't have audio but original videos do.

Workflow:
1. Read scene CSV files to get timecodes
2. Extract audio segments from original videos (set_1/)
3. Analyze each segment
4. Output per-scene audio data matching the portal format

Usage:
    python audio_from_originals.py
    python audio_from_originals.py --originals-dir /path/to/originals
    python audio_from_originals.py --max-scenes 50  # Limit for testing

Author: Pipeline Tools
"""

import os
import sys
import json
import csv
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import from existing audio modules
try:
    from audio_utils import extract_audio_robust, load_audio, AudioExtractionResult
    from robust_processor import IncrementalJSONWriter
except ImportError as e:
    print(f"Warning: Could not import audio modules: {e}")
    print("Some features may not be available.")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_ORIGINALS_DIR = SCRIPT_DIR.parent / "set_1"
DEFAULT_SCENES_DIR = SCRIPT_DIR.parent / "scenes"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "analysis"
AUDIO_TEMP_DIR = Path(tempfile.gettempdir()) / "audio_segments"

# Audio processing settings
TARGET_SAMPLE_RATE = 16000

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SceneTimecode:
    """Timecode information for a scene."""
    scene_number: int
    scene_name: str  # e.g., "VideoName-Scene-001"
    video_name: str  # Original video name
    start_seconds: float
    end_seconds: float
    duration: float


@dataclass
class AudioAnalysisResult:
    """Analysis result for a scene's audio."""
    scene_name: str
    scene_path: str
    audio_present: bool
    duration_seconds: float
    processing_status: str
    
    # Segmentation
    speech_ratio: float = 0.0
    non_verbal_ratio: float = 0.0
    silence_ratio: float = 0.0
    
    # Acoustic profile
    mean_pitch_hz: Optional[float] = None
    pitch_trend: str = "unknown"
    energy_trend: str = "unknown"
    
    # Classification
    dominant_cue: str = "unknown"
    valence_hint: str = "neutral"
    
    error: Optional[str] = None


# =============================================================================
# SCENE CSV PARSING
# =============================================================================

def parse_scene_csv(csv_path: Path) -> List[SceneTimecode]:
    """
    Parse a scene CSV file to extract timecodes.
    
    Args:
        csv_path: Path to the scene CSV file
        
    Returns:
        List of SceneTimecode objects
    """
    scenes = []
    video_name = csv_path.stem.replace('_scenes', '')
    
    try:
        with open(csv_path, 'r') as f:
            # Skip first line (timecode list)
            lines = f.readlines()
            if len(lines) < 2:
                return scenes
            
            # Find header line
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('Scene Number'):
                    header_idx = i
                    break
            
            # Parse CSV from header
            reader = csv.DictReader(lines[header_idx:])
            
            for row in reader:
                try:
                    scene_num = int(row.get('Scene Number', 0))
                    start_sec = float(row.get('Start Time (seconds)', 0))
                    end_sec = float(row.get('End Time (seconds)', 0))
                    duration = float(row.get('Length (seconds)', end_sec - start_sec))
                    
                    # Scene name matches the format used in scenes directory
                    scene_name = f"{video_name}-Scene-{scene_num:03d}"
                    
                    scenes.append(SceneTimecode(
                        scene_number=scene_num,
                        scene_name=scene_name,
                        video_name=video_name,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        duration=duration
                    ))
                except (ValueError, KeyError) as e:
                    continue
                    
    except Exception as e:
        print(f"  Warning: Could not parse {csv_path}: {e}")
    
    return scenes


def find_original_video(video_name: str, originals_dir: Path) -> Optional[Path]:
    """
    Find the original video file matching a video name.
    
    Args:
        video_name: Base name of the video (without extension)
        originals_dir: Directory containing original videos
        
    Returns:
        Path to the original video or None if not found
    """
    extensions = ['.mp4', '.MP4', '.avi', '.mkv', '.mov', '.webm', '.m4v']
    
    for ext in extensions:
        path = originals_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    
    # Try case-insensitive search
    for file in originals_dir.iterdir():
        if file.stem.lower() == video_name.lower() and file.suffix.lower() in [e.lower() for e in extensions]:
            return file
    
    return None


def extract_audio_segment(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path
) -> bool:
    """
    Extract an audio segment from a video using ffmpeg.
    
    Args:
        video_path: Path to the source video
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        output_path: Path for the output audio file
        
    Returns:
        True if extraction succeeded
    """
    duration = end_seconds - start_seconds
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_seconds),
        '-i', str(video_path),
        '-t', str(duration),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', str(TARGET_SAMPLE_RATE),
        '-ac', '1',  # Mono
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0 and output_path.exists()
    except Exception:
        return False


def check_video_has_audio(video_path: Path) -> bool:
    """Check if a video file has an audio stream."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return 'audio' in result.stdout.lower()
    except Exception:
        return False


# =============================================================================
# AUDIO ANALYSIS
# =============================================================================

class SimpleAudioAnalyzer:
    """
    Simplified audio analyzer for segment analysis.
    
    Uses basic signal processing when ML models aren't available.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.wav2vec2 = None
        self.clap = None
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Attempt to load audio models."""
        try:
            import torch
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            
            print("  Loading Wav2Vec2...")
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            if self.device == "cuda" and torch.cuda.is_available():
                self.wav2vec2 = self.wav2vec2.cuda()
            self.wav2vec2.eval()
            print("    Wav2Vec2 loaded")
        except Exception as e:
            print(f"  Warning: Could not load Wav2Vec2: {e}")
            self.wav2vec2 = None
    
    def analyze_segment(self, audio_path: Path) -> Dict:
        """
        Analyze an audio segment.
        
        Args:
            audio_path: Path to audio file (WAV)
            
        Returns:
            Dict with analysis results
        """
        import numpy as np
        
        result = {
            'audio_present': False,
            'duration_seconds': 0,
            'speech_ratio': 0,
            'silence_ratio': 1.0,
            'non_verbal_ratio': 0,
            'mean_pitch_hz': None,
            'pitch_trend': 'unknown',
            'energy_trend': 'unknown',
            'dominant_cue': 'silence',
            'valence_hint': 'neutral'
        }
        
        try:
            # Load audio
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            
            if len(audio) == 0:
                return result
            
            result['audio_present'] = True
            result['duration_seconds'] = len(audio) / sr
            
            # Basic energy analysis
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Compute RMS energy in windows
            window_size = int(0.1 * sr)  # 100ms windows
            hop_size = window_size // 2
            
            energies = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                energies.append(rms)
            
            if not energies:
                return result
            
            energies = np.array(energies)
            
            # Silence detection (low energy)
            silence_threshold = np.percentile(energies, 10) * 2
            silence_ratio = np.mean(energies < silence_threshold)
            result['silence_ratio'] = float(silence_ratio)
            
            # Energy trend
            if len(energies) > 2:
                first_half = np.mean(energies[:len(energies)//2])
                second_half = np.mean(energies[len(energies)//2:])
                if second_half > first_half * 1.2:
                    result['energy_trend'] = 'rising'
                elif second_half < first_half * 0.8:
                    result['energy_trend'] = 'falling'
                else:
                    result['energy_trend'] = 'stable'
            
            # Simple classification based on energy patterns
            mean_energy = np.mean(energies)
            energy_std = np.std(energies)
            
            if silence_ratio > 0.7:
                result['dominant_cue'] = 'silence'
                result['valence_hint'] = 'neutral'
            elif energy_std / (mean_energy + 1e-6) > 1.0:
                # High variance - likely speech or moaning
                result['dominant_cue'] = 'speech'
                result['speech_ratio'] = 1.0 - silence_ratio
                result['valence_hint'] = 'ambiguous'
            else:
                # Ambient noise
                result['dominant_cue'] = 'ambient'
                result['non_verbal_ratio'] = 1.0 - silence_ratio
                result['valence_hint'] = 'neutral'
            
        except Exception as e:
            result['error'] = str(e)
        
        return result


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_all_scenes(
    originals_dir: Path,
    scenes_dir: Path,
    output_dir: Path,
    max_scenes: Optional[int] = None,
    device: str = "cpu"
) -> Dict:
    """
    Process all scenes by extracting audio from original videos.
    
    Args:
        originals_dir: Directory with original videos
        scenes_dir: Directory with scene CSVs
        output_dir: Output directory for analysis
        max_scenes: Limit number of scenes (for testing)
        device: Device for ML models
        
    Returns:
        Summary statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all scene CSV files
    csv_files = sorted(scenes_dir.glob("*_scenes.csv"))
    print(f"Found {len(csv_files)} scene CSV files")
    
    # Parse all scenes
    all_scenes: List[SceneTimecode] = []
    videos_with_audio = set()
    
    for csv_file in csv_files:
        scenes = parse_scene_csv(csv_file)
        if scenes:
            # Check if original video has audio
            video_name = scenes[0].video_name
            original = find_original_video(video_name, originals_dir)
            
            if original and check_video_has_audio(original):
                all_scenes.extend(scenes)
                videos_with_audio.add(video_name)
            else:
                print(f"  Skipping {video_name}: no audio in original")
    
    print(f"Found {len(all_scenes)} scenes from {len(videos_with_audio)} videos with audio")
    
    if max_scenes:
        all_scenes = all_scenes[:max_scenes]
        print(f"Limited to {max_scenes} scenes")
    
    if not all_scenes:
        return {'total_scenes': 0, 'message': 'No scenes with audio found'}
    
    # Initialize analyzer
    print("\nInitializing audio analyzer...")
    analyzer = SimpleAudioAnalyzer(device=device)
    
    # Initialize incremental writer
    output_file = output_dir / "audio_analysis_pro.json"
    writer = IncrementalJSONWriter(output_file, backup_interval=10)
    writer.set_config({
        'source': 'original_videos',
        'originals_dir': str(originals_dir),
        'scenes_dir': str(scenes_dir)
    })
    
    # Process each scene
    successful = 0
    no_audio = 0
    failed = 0
    
    # Group scenes by video for efficiency
    scenes_by_video: Dict[str, List[SceneTimecode]] = {}
    for scene in all_scenes:
        if scene.video_name not in scenes_by_video:
            scenes_by_video[scene.video_name] = []
        scenes_by_video[scene.video_name].append(scene)
    
    for video_name, scenes in tqdm(scenes_by_video.items(), desc="Processing videos"):
        original = find_original_video(video_name, originals_dir)
        if not original:
            for scene in scenes:
                writer.add_error(scene.scene_name, 'no_original', f"Original video not found")
                failed += 1
            continue
        
        for scene in tqdm(scenes, desc=f"  {video_name}", leave=False):
            scene_path = str(scenes_dir / f"{scene.scene_name}.mp4")
            
            # Extract audio segment
            temp_audio = AUDIO_TEMP_DIR / f"{scene.scene_name}.wav"
            
            extracted = extract_audio_segment(
                original,
                scene.start_seconds,
                scene.end_seconds,
                temp_audio
            )
            
            if not extracted or not temp_audio.exists():
                result = AudioAnalysisResult(
                    scene_name=scene.scene_name,
                    scene_path=scene_path,
                    audio_present=False,
                    duration_seconds=scene.duration,
                    processing_status='no_audio'
                )
                no_audio += 1
            else:
                # Analyze the segment
                analysis = analyzer.analyze_segment(temp_audio)
                
                result = AudioAnalysisResult(
                    scene_name=scene.scene_name,
                    scene_path=scene_path,
                    audio_present=analysis.get('audio_present', False),
                    duration_seconds=analysis.get('duration_seconds', scene.duration),
                    processing_status='success',
                    speech_ratio=analysis.get('speech_ratio', 0),
                    non_verbal_ratio=analysis.get('non_verbal_ratio', 0),
                    silence_ratio=analysis.get('silence_ratio', 1.0),
                    mean_pitch_hz=analysis.get('mean_pitch_hz'),
                    pitch_trend=analysis.get('pitch_trend', 'unknown'),
                    energy_trend=analysis.get('energy_trend', 'unknown'),
                    dominant_cue=analysis.get('dominant_cue', 'unknown'),
                    valence_hint=analysis.get('valence_hint', 'neutral'),
                    error=analysis.get('error')
                )
                
                if result.audio_present:
                    successful += 1
                else:
                    no_audio += 1
                
                # Cleanup temp file
                temp_audio.unlink(missing_ok=True)
            
            # Convert to dict for JSON
            writer.add_analysis({
                'scene_path': result.scene_path,
                'scene_name': result.scene_name,
                'audio_present': result.audio_present,
                'duration_seconds': result.duration_seconds,
                'processing_status': result.processing_status,
                'segmentation': {
                    'speech_ratio': result.speech_ratio,
                    'non_verbal_ratio': result.non_verbal_ratio,
                    'silence_ratio': result.silence_ratio
                },
                'acoustic_profile': {
                    'mean_pitch_hz': result.mean_pitch_hz,
                    'pitch_trend': result.pitch_trend,
                    'energy_trend': result.energy_trend
                },
                'classification': {
                    'dominant_cue': result.dominant_cue,
                    'valence_hint': result.valence_hint
                },
                'error': result.error
            })
    
    # Update summary
    summary = {
        'total_scenes': successful + no_audio + failed,
        'successful': successful,
        'no_audio': no_audio,
        'failed': failed,
        'videos_processed': len(scenes_by_video)
    }
    writer.update_summary(summary)
    writer.finalize()
    
    print(f"\nSaved to {output_file}")
    print(f"  Successful: {successful}")
    print(f"  No audio: {no_audio}")
    print(f"  Failed: {failed}")
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(
        description="Extract and analyze audio from original videos using scene timecodes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--originals-dir', '-o', type=Path, default=DEFAULT_ORIGINALS_DIR,
                       help=f'Directory with original videos (default: {DEFAULT_ORIGINALS_DIR})')
    parser.add_argument('--scenes-dir', '-s', type=Path, default=DEFAULT_SCENES_DIR,
                       help=f'Directory with scene CSVs (default: {DEFAULT_SCENES_DIR})')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--max-scenes', '-m', type=int, default=None,
                       help='Maximum scenes to process (for testing)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for ML models (default: auto-detect)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Audio Analysis from Original Videos")
    print("=" * 70)
    print(f"Originals Dir: {args.originals_dir}")
    print(f"Scenes Dir:    {args.scenes_dir}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Device:        {args.device}")
    if args.max_scenes:
        print(f"Max Scenes:    {args.max_scenes}")
    print("=" * 70)
    
    if not args.originals_dir.exists():
        print(f"ERROR: Originals directory not found: {args.originals_dir}")
        return
    
    if not args.scenes_dir.exists():
        print(f"ERROR: Scenes directory not found: {args.scenes_dir}")
        return
    
    summary = process_all_scenes(
        originals_dir=args.originals_dir,
        scenes_dir=args.scenes_dir,
        output_dir=args.output_dir,
        max_scenes=args.max_scenes,
        device=args.device
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()
