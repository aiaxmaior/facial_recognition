#!/usr/bin/env python3
"""
Audio Processing Utilities
==========================

Shared utilities for robust audio extraction and validation:
- FFmpeg audio extraction with timeout protection
- Audio stream detection (ffprobe)
- Audio validation (corruption, duration, silence detection)
- Error classification and structured results

Used by audio_analyzer.py for both professional and hobby tracks.

CRITICAL: This script processes files WITHOUT interpreting content.
All operations are based on technical audio properties only.
"""

import json
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Try imports - graceful degradation if not installed
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default sample rate for all audio processing (Wav2Vec2/HuBERT standard)
DEFAULT_SAMPLE_RATE = 16000

# Minimum audio duration for model processing
MIN_AUDIO_DURATION = 0.5  # seconds

# Silence detection threshold (RMS energy)
SILENCE_THRESHOLD = 1e-6

# FFmpeg extraction timeout
DEFAULT_EXTRACTION_TIMEOUT = 60  # seconds

# Supported audio codecs for extraction
SUPPORTED_CODECS = {'aac', 'mp3', 'opus', 'vorbis', 'flac', 'pcm_s16le', 'pcm_s24le', 'ac3', 'eac3'}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AudioStreamInfo:
    """Information about an audio stream from ffprobe."""
    has_audio: bool
    codec_name: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration: Optional[float] = None
    bit_rate: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of audio validation."""
    valid: bool
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    rms_energy: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class AudioExtractionResult:
    """Result of audio extraction attempt."""
    success: bool
    wav_path: Optional[Path] = None
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stream_info: Optional[AudioStreamInfo] = None


@dataclass
class AudioData:
    """Loaded audio data ready for processing."""
    samples: np.ndarray  # Audio samples (mono, normalized)
    sample_rate: int
    duration_seconds: float
    source_path: str
    is_silent: bool = False
    rms_energy: float = 0.0


# =============================================================================
# FFPROBE - AUDIO STREAM DETECTION
# =============================================================================

def probe_audio_stream(video_path: Path, timeout: int = 10) -> AudioStreamInfo:
    """
    Check if video has audio stream using ffprobe.
    
    Args:
        video_path: Path to video file
        timeout: Timeout in seconds for ffprobe
        
    Returns:
        AudioStreamInfo with stream details or has_audio=False
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name,sample_rate,channels,duration,bit_rate',
        '-of', 'json',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if result.returncode != 0:
            return AudioStreamInfo(has_audio=False)
        
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        
        if not streams:
            return AudioStreamInfo(has_audio=False)
        
        stream = streams[0]
        return AudioStreamInfo(
            has_audio=True,
            codec_name=stream.get('codec_name'),
            sample_rate=int(stream.get('sample_rate', 0)) if stream.get('sample_rate') else None,
            channels=int(stream.get('channels', 0)) if stream.get('channels') else None,
            duration=float(stream.get('duration', 0)) if stream.get('duration') else None,
            bit_rate=int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None
        )
        
    except subprocess.TimeoutExpired:
        return AudioStreamInfo(has_audio=False)
    except (json.JSONDecodeError, KeyError, ValueError):
        return AudioStreamInfo(has_audio=False)


def get_video_duration(video_path: Path, timeout: int = 10) -> Optional[float]:
    """
    Get video duration using ffprobe (for fallback when no audio).
    
    Args:
        video_path: Path to video file
        timeout: Timeout in seconds
        
    Returns:
        Duration in seconds or None
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        data = json.loads(result.stdout)
        return float(data.get('format', {}).get('duration', 0))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None


# =============================================================================
# FFMPEG - AUDIO EXTRACTION
# =============================================================================

def extract_audio_ffmpeg(
    video_path: Path,
    output_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timeout: int = DEFAULT_EXTRACTION_TIMEOUT
) -> Tuple[bool, Optional[str]]:
    """
    Extract audio from video using FFmpeg.
    
    Args:
        video_path: Path to input video
        output_path: Path for output WAV file
        sample_rate: Target sample rate
        timeout: Timeout in seconds
        
    Returns:
        (success, error_message)
    """
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '1',  # Mono
        '-f', 'wav',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if result.returncode != 0:
            # Parse FFmpeg error
            stderr = result.stderr
            if 'does not contain any stream' in stderr:
                return False, "No audio stream in video"
            elif 'Invalid data found' in stderr:
                return False, "Invalid or corrupt audio data"
            elif 'Avi decoder' in stderr or 'codec' in stderr.lower():
                return False, f"Codec error: {stderr[-200:]}"
            else:
                return False, f"FFmpeg error: {stderr[-200:]}"
        
        # Verify output file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False, "FFmpeg produced empty output"
        
        return True, None
        
    except subprocess.TimeoutExpired:
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        return False, f"FFmpeg timed out after {timeout}s"
    except Exception as e:
        return False, f"FFmpeg exception: {str(e)}"


# =============================================================================
# AUDIO VALIDATION
# =============================================================================

def validate_audio(wav_path: Path) -> ValidationResult:
    """
    Validate audio file integrity.
    
    Checks:
    - File is readable
    - No NaN or Inf values
    - Duration >= MIN_AUDIO_DURATION
    - Not completely silent
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        ValidationResult with validation status
    """
    if not SOUNDFILE_AVAILABLE and not LIBROSA_AVAILABLE:
        return ValidationResult(
            valid=False,
            error_type='missing_dependency',
            error_message='Neither soundfile nor librosa available'
        )
    
    try:
        # Try soundfile first (faster)
        if SOUNDFILE_AVAILABLE:
            audio, sr = sf.read(str(wav_path))
        elif LIBROSA_AVAILABLE:
            audio, sr = librosa.load(str(wav_path), sr=None)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Check for NaN/Inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return ValidationResult(
                valid=False,
                error_type='audio_corrupt',
                error_message='Audio contains NaN or Inf values'
            )
        
        # Check duration
        duration = len(audio) / sr
        if duration < MIN_AUDIO_DURATION:
            return ValidationResult(
                valid=False,
                duration=duration,
                sample_rate=sr,
                error_type='audio_too_short',
                error_message=f'Duration {duration:.2f}s < {MIN_AUDIO_DURATION}s minimum'
            )
        
        # Check for silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < SILENCE_THRESHOLD:
            return ValidationResult(
                valid=True,  # Silent audio is valid, just flagged
                duration=duration,
                sample_rate=sr,
                rms_energy=rms,
                error_type='audio_silent',
                error_message=f'RMS energy {rms:.2e} below threshold'
            )
        
        return ValidationResult(
            valid=True,
            duration=duration,
            sample_rate=sr,
            channels=1,  # We convert to mono
            rms_energy=rms
        )
        
    except Exception as e:
        return ValidationResult(
            valid=False,
            error_type='audio_corrupt',
            error_message=f'Failed to read audio: {str(e)}'
        )


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_audio_robust(
    video_path: Path,
    output_dir: Optional[Path] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timeout_seconds: int = DEFAULT_EXTRACTION_TIMEOUT,
    cleanup_on_error: bool = True
) -> AudioExtractionResult:
    """
    Extract audio with comprehensive error handling.
    
    Steps:
    1. Check if video has audio stream (ffprobe)
    2. Extract with timeout protection (ffmpeg)
    3. Validate extracted audio
    4. Return structured result
    
    Args:
        video_path: Path to video file
        output_dir: Directory for output WAV (default: temp directory)
        sample_rate: Target sample rate
        timeout_seconds: FFmpeg timeout
        cleanup_on_error: Remove output file on validation failure
        
    Returns:
        AudioExtractionResult with extraction status and details
    """
    video_path = Path(video_path)
    
    # Check video file exists
    if not video_path.exists():
        return AudioExtractionResult(
            success=False,
            error_type='file_not_found',
            error_message=f'Video file does not exist: {video_path}'
        )
    
    if video_path.stat().st_size == 0:
        return AudioExtractionResult(
            success=False,
            error_type='empty_file',
            error_message=f'Video file is empty: {video_path}'
        )
    
    # Step 1: Probe for audio stream
    stream_info = probe_audio_stream(video_path)
    
    if not stream_info.has_audio:
        return AudioExtractionResult(
            success=False,
            error_type='no_audio_stream',
            error_message='Video contains no audio track',
            stream_info=stream_info
        )
    
    # Step 2: Determine output path
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav_path = output_dir / f"{video_path.stem}.wav"
    
    # Step 3: Extract audio
    success, error_msg = extract_audio_ffmpeg(
        video_path,
        wav_path,
        sample_rate=sample_rate,
        timeout=timeout_seconds
    )
    
    if not success:
        error_type = 'extraction_failed'
        if 'timed out' in (error_msg or '').lower():
            error_type = 'extraction_timeout'
        elif 'codec' in (error_msg or '').lower():
            error_type = 'codec_unsupported'
        
        return AudioExtractionResult(
            success=False,
            error_type=error_type,
            error_message=error_msg,
            stream_info=stream_info
        )
    
    # Step 4: Validate extracted audio
    validation = validate_audio(wav_path)
    
    if not validation.valid:
        if cleanup_on_error:
            wav_path.unlink(missing_ok=True)
        return AudioExtractionResult(
            success=False,
            error_type=validation.error_type,
            error_message=validation.error_message,
            stream_info=stream_info
        )
    
    return AudioExtractionResult(
        success=True,
        wav_path=wav_path,
        duration_seconds=validation.duration,
        sample_rate=validation.sample_rate,
        channels=validation.channels,
        stream_info=stream_info
    )


# =============================================================================
# AUDIO LOADING
# =============================================================================

def load_audio(
    wav_path: Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True
) -> Optional[AudioData]:
    """
    Load audio file into memory.
    
    Args:
        wav_path: Path to WAV file
        target_sr: Target sample rate (will resample if needed)
        normalize: Normalize to [-1, 1] range
        
    Returns:
        AudioData or None on failure
    """
    wav_path = Path(wav_path)
    
    if not wav_path.exists():
        return None
    
    try:
        if LIBROSA_AVAILABLE:
            # librosa handles resampling automatically
            audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
        elif SOUNDFILE_AVAILABLE:
            audio, sr = sf.read(str(wav_path))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            # Manual resampling would be needed here if sr != target_sr
        else:
            return None
        
        # Normalize
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        
        # Compute RMS
        rms = np.sqrt(np.mean(audio ** 2))
        is_silent = rms < SILENCE_THRESHOLD
        
        return AudioData(
            samples=audio,
            sample_rate=sr,
            duration_seconds=len(audio) / sr,
            source_path=str(wav_path),
            is_silent=is_silent,
            rms_energy=rms
        )
        
    except Exception as e:
        print(f"Error loading audio {wav_path}: {e}")
        return None


# =============================================================================
# SCENE FILTERING
# =============================================================================

def get_scenes_with_audio(
    scenes_dir: Path,
    extensions: set = {'.mp4', '.MP4', '.avi', '.mkv'}
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Filter scene files to those with audio streams.
    
    Args:
        scenes_dir: Directory containing scene clips
        extensions: Video file extensions to check
        
    Returns:
        (scenes_with_audio, scenes_without_audio_with_reason)
    """
    scenes_dir = Path(scenes_dir)
    
    if not scenes_dir.exists():
        return [], []
    
    all_scenes = sorted([
        f for f in scenes_dir.iterdir()
        if f.suffix in extensions
    ])
    
    with_audio = []
    without_audio = []
    
    for scene in all_scenes:
        info = probe_audio_stream(scene)
        if info.has_audio:
            with_audio.append(scene)
        else:
            without_audio.append((scene, 'no_audio_stream'))
    
    return with_audio, without_audio


def get_scenes_for_processing(
    scenes_dir: Path,
    detections_file: Optional[Path] = None,
    filter_person_present: bool = True
) -> List[Path]:
    """
    Get list of scenes to process, optionally filtered by detection results.
    
    Args:
        scenes_dir: Directory containing scene clips
        detections_file: Optional path to detections.json for filtering
        filter_person_present: If True and detections available, only process
                               scenes with person_present=True
                               
    Returns:
        List of scene paths to process
    """
    scenes_dir = Path(scenes_dir)
    
    # Get all scenes with audio
    scenes_with_audio, _ = get_scenes_with_audio(scenes_dir)
    
    if not filter_person_present or detections_file is None:
        return scenes_with_audio
    
    detections_file = Path(detections_file)
    if not detections_file.exists():
        return scenes_with_audio
    
    # Load detections and filter
    try:
        with open(detections_file) as f:
            detections = json.load(f)
        
        person_present_paths = set()
        for analysis in detections.get('analyses', []):
            if analysis.get('person_present', False):
                person_present_paths.add(analysis.get('scene_path', ''))
        
        filtered = []
        for scene in scenes_with_audio:
            # Check if scene path matches (may need to handle relative vs absolute)
            if str(scene) in person_present_paths or scene.name in [Path(p).name for p in person_present_paths]:
                filtered.append(scene)
        
        return filtered if filtered else scenes_with_audio
        
    except (json.JSONDecodeError, KeyError):
        return scenes_with_audio


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def cleanup_temp_audio(output_dir: Path, keep_list: Optional[set] = None):
    """
    Clean up temporary audio files.
    
    Args:
        output_dir: Directory containing WAV files
        keep_list: Set of filenames to keep (optional)
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    
    for wav_file in output_dir.glob("*.wav"):
        if keep_list is None or wav_file.name not in keep_list:
            wav_file.unlink(missing_ok=True)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test audio utilities")
    parser.add_argument('video', type=Path, nargs='?', help='Video file to test')
    parser.add_argument('--probe-only', action='store_true', help='Only probe, do not extract')
    args = parser.parse_args()
    
    if args.video:
        print(f"Testing audio utilities on: {args.video}")
        print("=" * 60)
        
        # Probe
        print("\n1. Probing audio stream...")
        info = probe_audio_stream(args.video)
        print(f"   Has audio: {info.has_audio}")
        if info.has_audio:
            print(f"   Codec: {info.codec_name}")
            print(f"   Sample rate: {info.sample_rate}")
            print(f"   Channels: {info.channels}")
            print(f"   Duration: {info.duration}s")
        
        if not args.probe_only and info.has_audio:
            # Extract
            print("\n2. Extracting audio...")
            result = extract_audio_robust(args.video)
            print(f"   Success: {result.success}")
            if result.success:
                print(f"   WAV path: {result.wav_path}")
                print(f"   Duration: {result.duration_seconds:.2f}s")
                print(f"   Sample rate: {result.sample_rate}")
                
                # Load
                print("\n3. Loading audio...")
                audio = load_audio(result.wav_path)
                if audio:
                    print(f"   Samples: {len(audio.samples)}")
                    print(f"   RMS energy: {audio.rms_energy:.4f}")
                    print(f"   Silent: {audio.is_silent}")
                
                # Cleanup
                result.wav_path.unlink(missing_ok=True)
            else:
                print(f"   Error type: {result.error_type}")
                print(f"   Error message: {result.error_message}")
    else:
        print("Audio utilities module loaded successfully.")
        print(f"  LIBROSA_AVAILABLE: {LIBROSA_AVAILABLE}")
        print(f"  SOUNDFILE_AVAILABLE: {SOUNDFILE_AVAILABLE}")
        print(f"  DEFAULT_SAMPLE_RATE: {DEFAULT_SAMPLE_RATE}")
        print(f"  MIN_AUDIO_DURATION: {MIN_AUDIO_DURATION}s")
