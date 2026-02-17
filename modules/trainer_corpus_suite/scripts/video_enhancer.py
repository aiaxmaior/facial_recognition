#!/usr/bin/env python3
"""
Video Enhancer Module (Optional)
================================

Enhances video quality through denoising, sharpening, and optional upscaling.
Designed as an optional preprocessing step for low-quality source footage.

Enhancement Modes:
- 'light': FFmpeg filters only (hqdn3d + unsharp) - fastest, no deps
- 'medium': OpenCV DNN super-resolution (ESPCN/FSRCNN) - balanced
- 'heavy': Real-ESRGAN-ncnn-vulkan - best quality, requires binary

Usage:
    # Enhance single video
    python video_enhancer.py input.mp4 -o output.mp4 --mode light
    
    # Batch enhance directory
    python video_enhancer.py --input-dir processed/ --output-dir enhanced/ --mode medium
    
    # Quality assessment only (no processing)
    python video_enhancer.py --assess-only processed/

CRITICAL: This script processes files WITHOUT interpreting visual content.
All operations are based on automated quality metrics only.
"""

import subprocess
import sys
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

class EnhanceMode(Enum):
    """Enhancement intensity levels."""
    LIGHT = "light"    # FFmpeg filters only
    MEDIUM = "medium"  # OpenCV DNN upscaling
    HEAVY = "heavy"    # Real-ESRGAN (if available)


# FFmpeg enhancement filter chains
# - hqdn3d: High-quality 3D denoiser (spatial + temporal)
# - unsharp: Sharpening/unsharp mask filter
FILTER_PRESETS = {
    # Light denoising + subtle sharpening (for minor artifacts)
    'light': 'hqdn3d=2:1.5:3:2.5,unsharp=3:3:0.5:3:3:0',
    
    # Medium denoising + moderate sharpening (for compressed video)
    'medium': 'hqdn3d=4:3:6:4.5,unsharp=5:5:0.8:3:3:0',
    
    # Heavy denoising + strong sharpening (for very poor quality)
    'heavy': 'hqdn3d=6:4:8:6,unsharp=5:5:1.2:5:5:0.5',
}

# Quality thresholds for auto-detection (based on metadata only)
QUALITY_THRESHOLDS = {
    'bitrate_low': 500_000,      # Below 500kbps = definitely needs enhancement
    'bitrate_medium': 1_500_000, # Below 1.5Mbps = might benefit
}

# OpenCV super-resolution models (small, fast)
OPENCV_SR_MODELS = {
    'espcn_x2': {
        'url': 'https://github.com/Saafke/ESPCN_Tensorflow/raw/master/export/ESPCN_x2.pb',
        'scale': 2,
        'name': 'espcn'
    },
    'espcn_x3': {
        'url': 'https://github.com/Saafke/ESPCN_Tensorflow/raw/master/export/ESPCN_x3.pb',
        'scale': 3,
        'name': 'espcn'
    },
    'fsrcnn_x2': {
        'url': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/export/FSRCNN_x2.pb',
        'scale': 2,
        'name': 'fsrcnn'
    },
}

# Default paths
MODELS_DIR = Path(__file__).parent / "sr_models"
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "processed"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "enhanced"

VIDEO_EXTENSIONS = {'.mp4', '.MP4', '.avi', '.mkv', '.flv', '.m4v', '.mov', '.webm'}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QualityMetrics:
    """Video quality assessment metrics (no visual inspection required)."""
    path: Path
    bitrate: int           # bits per second
    resolution: Tuple[int, int]
    codec: str
    estimated_quality: str  # 'good', 'medium', 'poor'
    enhancement_recommended: bool
    suggested_mode: EnhanceMode


# =============================================================================
# QUALITY ASSESSMENT
# =============================================================================

def assess_video_quality(video_path: Path) -> Optional[QualityMetrics]:
    """
    Assess video quality using metadata only (no visual analysis).
    
    Quality indicators examined:
    - Bitrate: Low bitrate often correlates with compression artifacts
    - Codec: Older codecs (mpeg4, wmv) typically produce more artifacts
    - Resolution vs file size: High res but small file = over-compressed
    
    Args:
        video_path: Path to video file
        
    Returns:
        QualityMetrics dataclass or None if assessment fails
    """
    try:
        # Get video metadata via ffprobe
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
        
        # Extract metrics from ffprobe output
        video_stream = next(
            (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
            {}
        )
        format_info = data.get('format', {})
        
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        codec = video_stream.get('codec_name', 'unknown')
        bitrate = int(format_info.get('bit_rate', 0))
        
        # Estimate quality based on metadata thresholds
        quality = 'good'
        enhance = False
        mode = EnhanceMode.LIGHT
        
        # Check bitrate thresholds
        if bitrate < QUALITY_THRESHOLDS['bitrate_low']:
            quality = 'poor'
            enhance = True
            mode = EnhanceMode.HEAVY
        elif bitrate < QUALITY_THRESHOLDS['bitrate_medium']:
            quality = 'medium'
            enhance = True
            mode = EnhanceMode.MEDIUM
            
        # Check codec - older codecs often have more artifacts
        legacy_codecs = ('mpeg4', 'wmv2', 'wmv3', 'flv1', 'h263', 'mpeg2video')
        if codec in legacy_codecs:
            quality = 'medium' if quality == 'good' else quality
            enhance = True
            mode = EnhanceMode.MEDIUM if mode == EnhanceMode.LIGHT else mode
            
        return QualityMetrics(
            path=video_path,
            bitrate=bitrate,
            resolution=(width, height),
            codec=codec,
            estimated_quality=quality,
            enhancement_recommended=enhance,
            suggested_mode=mode
        )
        
    except Exception as e:
        print(f"  Warning: Could not assess {video_path.name}: {e}")
        return None


# =============================================================================
# ENHANCEMENT METHODS
# =============================================================================

def enhance_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    filter_preset: str = 'medium'
) -> bool:
    """
    Apply FFmpeg-based enhancement (denoising + sharpening).
    
    This is the lightest option - uses built-in FFmpeg filters:
    - hqdn3d: High-quality 3D denoiser (spatial + temporal)
    - unsharp: Sharpening/unsharp mask filter
    
    Args:
        input_path: Source video path
        output_path: Destination path
        filter_preset: 'light', 'medium', or 'heavy'
        
    Returns:
        True if enhancement succeeded
    """
    filter_chain = FILTER_PRESETS.get(filter_preset, FILTER_PRESETS['medium'])
    
    cmd = [
        'ffmpeg', '-y', 
        '-i', str(input_path),
        '-vf', filter_chain,
        '-c:v', 'libx264', 
        '-crf', '18',           # High quality
        '-preset', 'medium',    # Balance speed/compression
        '-an',                  # Strip audio
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=600  # 10 minute timeout per video
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Timeout enhancing {input_path.name}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def ensure_sr_model(model_name: str) -> Optional[Path]:
    """
    Download super-resolution model if not present locally.
    
    Args:
        model_name: Key from OPENCV_SR_MODELS dict
        
    Returns:
        Path to model file or None on failure
    """
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_info = OPENCV_SR_MODELS.get(model_name)
    if not model_info:
        print(f"  Unknown model: {model_name}")
        return None
        
    model_path = MODELS_DIR / f"{model_name}.pb"
    
    if not model_path.exists():
        print(f"  Downloading {model_name} model...")
        try:
            import urllib.request
            urllib.request.urlretrieve(model_info['url'], model_path)
            print(f"  Downloaded to {model_path}")
        except Exception as e:
            print(f"  Failed to download model: {e}")
            return None
            
    return model_path


def enhance_with_opencv_sr(
    input_path: Path,
    output_path: Path,
    model_name: str = 'espcn_x2'
) -> bool:
    """
    Apply OpenCV DNN super-resolution.
    
    Uses lightweight neural network models (ESPCN/FSRCNN) for upscaling.
    Processes frame-by-frame with optional GPU acceleration via CUDA.
    
    Args:
        input_path: Source video path
        output_path: Destination path  
        model_name: 'espcn_x2', 'espcn_x3', or 'fsrcnn_x2'
        
    Returns:
        True if enhancement succeeded
    """
    try:
        import cv2
        
        # Check if dnn_superres is available (requires opencv-contrib-python)
        if not hasattr(cv2, 'dnn_superres'):
            print("  Warning: cv2.dnn_superres not available.")
            print("  Install with: pip install opencv-contrib-python")
            return False
            
        # Ensure model is downloaded
        model_path = ensure_sr_model(model_name)
        if model_path is None:
            return False
            
        # Initialize super-resolution module
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        
        model_info = OPENCV_SR_MODELS[model_name]
        sr.setModel(model_info['name'], model_info['scale'])
        
        # Try to use CUDA if available for GPU acceleration
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("  Using CUDA acceleration")
        except:
            pass  # Fall back to CPU
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate output dimensions
        scale = model_info['scale']
        out_width = width * scale
        out_height = height * scale
        
        # Initialize video writer (temp file, will re-encode)
        temp_path = output_path.with_suffix('.temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_path), fourcc, fps, (out_width, out_height))
        
        # Process frames with progress bar
        with tqdm(total=frame_count, desc=f"  Upscaling", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Apply super-resolution
                upscaled = sr.upsample(frame)
                out.write(upscaled)
                pbar.update(1)
        
        cap.release()
        out.release()
        
        # Re-encode with x264 for better compression
        subprocess.run([
            'ffmpeg', '-y', 
            '-i', str(temp_path),
            '-c:v', 'libx264', 
            '-crf', '18', 
            '-preset', 'medium',
            '-an',
            str(output_path)
        ], capture_output=True)
        
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
        return True
        
    except Exception as e:
        print(f"  Error in OpenCV SR: {e}")
        return False


def check_realesrgan_available() -> Optional[Path]:
    """
    Check if Real-ESRGAN binary is installed and available.
    
    Checks common installation locations and PATH.
    
    Returns:
        Path to binary or None if not found
    """
    # Check common installation locations
    locations = [
        Path(__file__).parent / "realesrgan-ncnn-vulkan",
        Path.home() / ".local" / "bin" / "realesrgan-ncnn-vulkan",
        Path("/usr/local/bin/realesrgan-ncnn-vulkan"),
    ]
    
    for loc in locations:
        if loc.exists() and loc.is_file():
            return loc
            
    # Check if in system PATH
    which_result = shutil.which("realesrgan-ncnn-vulkan")
    if which_result:
        return Path(which_result)
        
    return None


def enhance_with_realesrgan(
    input_path: Path,
    output_path: Path,
    model: str = 'realesrgan-x4plus'
) -> bool:
    """
    Apply Real-ESRGAN enhancement (requires binary installation).
    
    Best quality option but requires external binary.
    Download from: https://github.com/xinntao/Real-ESRGAN/releases
    
    Available models:
    - realesrgan-x4plus: General purpose (recommended)
    - realesr-animevideov3: Optimized for animation
    
    Args:
        input_path: Source video path
        output_path: Destination path
        model: Model name to use
        
    Returns:
        True if enhancement succeeded
    """
    binary = check_realesrgan_available()
    
    if binary is None:
        print("  Real-ESRGAN binary not found. Install from:")
        print("  https://github.com/xinntao/Real-ESRGAN/releases")
        return False
        
    cmd = [
        str(binary),
        '-i', str(input_path),
        '-o', str(output_path),
        '-n', model,
        '-f', 'mp4'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Timeout (30min) enhancing {input_path.name}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def enhance_video(
    input_path: Path,
    output_path: Path,
    mode: EnhanceMode = EnhanceMode.LIGHT
) -> bool:
    """
    Enhance a single video using the specified mode.
    
    Falls back to lighter modes if heavier modes fail or are unavailable.
    
    Fallback chain: heavy -> medium -> light
    
    Args:
        input_path: Source video path
        output_path: Destination path
        mode: Enhancement intensity level
        
    Returns:
        True if enhancement succeeded (possibly with fallback)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == EnhanceMode.HEAVY:
        # Try Real-ESRGAN first, fall back to medium if unavailable
        if check_realesrgan_available():
            print(f"  Using Real-ESRGAN (heavy mode)")
            if enhance_with_realesrgan(input_path, output_path):
                return True
        print("  Falling back to medium mode...")
        mode = EnhanceMode.MEDIUM
        
    if mode == EnhanceMode.MEDIUM:
        # Try OpenCV SR, fall back to light if unavailable
        print(f"  Using OpenCV Super-Resolution (medium mode)")
        if enhance_with_opencv_sr(input_path, output_path):
            return True
        print("  Falling back to light mode...")
        mode = EnhanceMode.LIGHT
        
    # Light mode - FFmpeg filters (always available)
    print(f"  Using FFmpeg filters (light mode)")
    return enhance_with_ffmpeg(input_path, output_path, 'medium')


def batch_enhance(
    input_dir: Path,
    output_dir: Path,
    mode: EnhanceMode = EnhanceMode.LIGHT,
    auto_detect: bool = True
) -> Dict:
    """
    Batch enhance all videos in a directory.
    
    Args:
        input_dir: Directory containing source videos
        output_dir: Directory for enhanced output
        mode: Default enhancement mode (can be overridden by auto-detect)
        auto_detect: If True, automatically select mode based on quality assessment
        
    Returns:
        Summary statistics dict with processed/skipped/failed counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    videos = sorted([
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in {e.lower() for e in VIDEO_EXTENSIONS}
    ])
    
    # Print configuration header
    print(f"\n{'='*60}")
    print(f"Video Enhancement Pipeline")
    print(f"{'='*60}")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Mode:       {mode.value} {'(auto-detect enabled)' if auto_detect else ''}")
    print(f"Found:      {len(videos)} videos")
    print(f"{'='*60}\n")
    
    results = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'details': []
    }
    
    for video_path in tqdm(videos, desc="Enhancing videos"):
        output_path = output_dir / f"{video_path.stem}_enhanced{video_path.suffix}"
        
        # Assess quality if auto-detect enabled
        effective_mode = mode
        if auto_detect:
            metrics = assess_video_quality(video_path)
            if metrics and not metrics.enhancement_recommended:
                print(f"\n  Skipping {video_path.name} - quality is already good")
                results['skipped'] += 1
                results['details'].append({
                    'file': video_path.name,
                    'status': 'skipped',
                    'reason': 'quality_good'
                })
                # Copy without enhancement
                shutil.copy2(video_path, output_path)
                continue
            elif metrics:
                effective_mode = metrics.suggested_mode
                print(f"\n  {video_path.name}: quality={metrics.estimated_quality}, "
                      f"bitrate={metrics.bitrate//1000}kbps, mode={effective_mode.value}")
        
        # Perform enhancement
        success = enhance_video(video_path, output_path, effective_mode)
        
        if success:
            results['processed'] += 1
            results['details'].append({
                'file': video_path.name,
                'status': 'enhanced',
                'mode': effective_mode.value
            })
        else:
            results['failed'] += 1
            results['details'].append({
                'file': video_path.name,
                'status': 'failed'
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Enhancement Complete")
    print(f"{'='*60}")
    print(f"Processed: {results['processed']}")
    print(f"Skipped:   {results['skipped']}")
    print(f"Failed:    {results['failed']}")
    print(f"{'='*60}\n")
    
    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for the video enhancer."""
    parser = argparse.ArgumentParser(
        description="Video Enhancement Module - Denoise, sharpen, and optionally upscale videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance single video with light denoising
  python video_enhancer.py video.mp4 -o enhanced.mp4 --mode light
  
  # Batch enhance with auto quality detection
  python video_enhancer.py --input-dir processed/ --output-dir enhanced/ --auto
  
  # Assess quality only (no processing)
  python video_enhancer.py --assess-only processed/
  
Enhancement Modes:
  light  - FFmpeg filters only (hqdn3d + unsharp) - fastest
  medium - OpenCV DNN super-resolution (ESPCN) - balanced
  heavy  - Real-ESRGAN upscaling - best quality, slowest
        """
    )
    
    parser.add_argument(
        'input', nargs='?', type=Path,
        help="Input video file (for single-file mode)"
    )
    
    parser.add_argument(
        '-o', '--output', type=Path,
        help="Output path (for single-file mode)"
    )
    
    parser.add_argument(
        '--input-dir', '-i', type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory for batch processing (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        '--output-dir', type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for batch processing (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        '--mode', '-m', 
        choices=['light', 'medium', 'heavy'],
        default='light',
        help="Enhancement mode (default: light)"
    )
    
    parser.add_argument(
        '--auto', action='store_true',
        help="Auto-detect quality and select appropriate enhancement mode"
    )
    
    parser.add_argument(
        '--assess-only', type=Path, metavar='DIR',
        help="Only assess video quality without processing"
    )
    
    args = parser.parse_args()
    
    # Quality assessment only mode
    if args.assess_only:
        assess_dir = args.assess_only
        videos = sorted([
            f for f in assess_dir.iterdir() 
            if f.suffix.lower() in {e.lower() for e in VIDEO_EXTENSIONS}
        ])
        
        print(f"\nQuality Assessment Report")
        print(f"{'='*90}")
        print(f"{'File':<35} {'Resolution':<12} {'Bitrate':<10} {'Codec':<10} {'Quality':<8} {'Enhance?'}")
        print(f"{'-'*90}")
        
        for video in videos:
            metrics = assess_video_quality(video)
            if metrics:
                res = f"{metrics.resolution[0]}x{metrics.resolution[1]}"
                br = f"{metrics.bitrate // 1000}kbps"
                enhance = f"Yes ({metrics.suggested_mode.value})" if metrics.enhancement_recommended else "No"
                print(f"{video.name:<35} {res:<12} {br:<10} {metrics.codec:<10} "
                      f"{metrics.estimated_quality:<8} {enhance}")
        
        print(f"{'='*90}\n")
        return
    
    # Single file mode
    if args.input and args.input.is_file():
        output = args.output or args.input.with_stem(f"{args.input.stem}_enhanced")
        mode = EnhanceMode(args.mode)
        
        print(f"\nEnhancing: {args.input}")
        print(f"Output:    {output}")
        print(f"Mode:      {mode.value}\n")
        
        success = enhance_video(args.input, output, mode)
        print(f"\n{'Success' if success else 'Failed'}!")
        sys.exit(0 if success else 1)
    
    # Batch mode (default)
    mode = EnhanceMode(args.mode)
    batch_enhance(args.input_dir, args.output_dir, mode, args.auto)


if __name__ == "__main__":
    main()
