#!/usr/bin/env python3
"""
Video Preprocessor for Training Data
=====================================

Standardizes videos to a consistent format for VLM and downstream model processing:
- Resolution: 960x544 (preserves aspect ratio, center-crops if needed)
- Codec: H.264 (libx264), CRF 18
- Frame rate: 25 fps
- Audio: Preserved (AAC codec for compatibility)
- Container: MP4

CRITICAL: This script processes files WITHOUT interpreting visual content.
All operations are based on metadata and automated transformations only.

Usage:
    python video_preprocessor.py [--input-dir DIR] [--output-dir DIR]
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Target output specifications for VLM analysis
TARGET_WIDTH = 960
TARGET_HEIGHT = 544
TARGET_FPS = 25
TARGET_CRF = 18  # Quality setting (lower = higher quality, 18 is visually lossless)

# Supported input formats
VIDEO_EXTENSIONS = {'.mp4', '.MP4', '.avi', '.mkv', '.flv', '.m4v', '.mov', '.webm'}

# Default paths (relative to training_data/)
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "set_1"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "processed"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VideoInfo:
    """Metadata extracted from a video file via ffprobe."""
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    size_bytes: int
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def frame_count(self) -> int:
        return int(self.duration * self.fps)


# =============================================================================
# VIDEO ANALYSIS (ffprobe)
# =============================================================================

def get_video_info(video_path: Path) -> Optional[VideoInfo]:
    """
    Extract video metadata using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        VideoInfo object or None if extraction fails
    """
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',  # First video stream only
        '-show_entries', 'stream=width,height,r_frame_rate,codec_name',
        '-show_entries', 'format=duration,size',
        '-of', 'json',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
        
        # Extract stream info
        stream = data.get('streams', [{}])[0]
        format_info = data.get('format', {})
        
        # Parse frame rate (can be "30/1" or "30000/1001")
        fps_str = stream.get('r_frame_rate', '25/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 25.0
        else:
            fps = float(fps_str)
        
        return VideoInfo(
            path=video_path,
            width=int(stream.get('width', 0)),
            height=int(stream.get('height', 0)),
            fps=fps,
            duration=float(format_info.get('duration', 0)),
            codec=stream.get('codec_name', 'unknown'),
            size_bytes=int(format_info.get('size', 0))
        )
        
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        return None


# =============================================================================
# VIDEO CONVERSION (ffmpeg)
# =============================================================================

def calculate_scale_filter(src_width: int, src_height: int, 
                           dst_width: int, dst_height: int) -> str:
    """
    Calculate ffmpeg filter chain to resize and crop video.
    
    Strategy:
    1. Scale to fit within target dimensions (preserve aspect ratio)
    2. Pad to target dimensions if needed (letterbox/pillarbox)
    3. Center crop to exact target dimensions
    
    Args:
        src_width, src_height: Source dimensions
        dst_width, dst_height: Target dimensions
        
    Returns:
        ffmpeg filter string
    """
    src_aspect = src_width / src_height
    dst_aspect = dst_width / dst_height
    
    if abs(src_aspect - dst_aspect) < 0.01:
        # Aspect ratios match closely - simple scale
        return f"scale={dst_width}:{dst_height}"
    
    # Scale to fill target (may overflow one dimension), then center crop
    # This preserves more content than letterboxing
    if src_aspect > dst_aspect:
        # Source is wider - scale by height, crop width
        scale_filter = f"scale=-2:{dst_height}"
    else:
        # Source is taller - scale by width, crop height
        scale_filter = f"scale={dst_width}:-2"
    
    # Center crop to exact dimensions
    crop_filter = f"crop={dst_width}:{dst_height}"
    
    return f"{scale_filter},{crop_filter}"


def convert_video(input_path: Path, output_path: Path, 
                  video_info: VideoInfo) -> Tuple[bool, str]:
    """
    Convert a video to target specifications using ffmpeg.
    
    Args:
        input_path: Source video path
        output_path: Destination path
        video_info: Metadata about source video
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Build filter chain
    scale_filter = calculate_scale_filter(
        video_info.width, video_info.height,
        TARGET_WIDTH, TARGET_HEIGHT
    )
    
    # Add FPS filter if needed
    if abs(video_info.fps - TARGET_FPS) > 0.1:
        video_filter = f"{scale_filter},fps={TARGET_FPS}"
    else:
        video_filter = scale_filter
    
    # Build ffmpeg command - preserve audio if present
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', str(input_path),
        '-vf', video_filter,
        '-c:v', 'libx264',
        '-crf', str(TARGET_CRF),
        '-preset', 'medium',  # Balance speed/compression
        '-pix_fmt', 'yuv420p',  # Maximum compatibility
        '-c:a', 'aac',  # Re-encode audio to AAC for compatibility
        '-b:a', '128k',  # Audio bitrate
        '-movflags', '+faststart',  # Enable streaming
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 min timeout for long videos
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr[:500]  # Truncate error
            
    except subprocess.TimeoutExpired:
        return False, "Timeout (>10 min)"
    except Exception as e:
        return False, str(e)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def discover_videos(input_dir: Path) -> List[Path]:
    """Find all video files in input directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_dir.glob(f"*{ext}"))
    return sorted(videos)


def process_directory(input_dir: Path, output_dir: Path) -> Dict:
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Directory containing source videos
        output_dir: Directory for processed videos
        
    Returns:
        Summary statistics dictionary
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover videos
    videos = discover_videos(input_dir)
    print(f"Found {len(videos)} videos in {input_dir}")
    
    # Track results
    results = {
        'total': len(videos),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    # Process each video
    for video_path in tqdm(videos, desc="Processing videos"):
        # Get video info
        info = get_video_info(video_path)
        
        if info is None:
            results['skipped'] += 1
            results['errors'].append({
                'file': video_path.name,
                'error': 'Could not read video metadata'
            })
            continue
        
        # Skip if dimensions are zero (corrupt file)
        if info.width == 0 or info.height == 0:
            results['skipped'] += 1
            results['errors'].append({
                'file': video_path.name,
                'error': 'Invalid dimensions (corrupt file?)'
            })
            continue
        
        # Generate output path (always .mp4)
        output_path = output_dir / f"{video_path.stem}.mp4"
        
        # Convert
        success, message = convert_video(video_path, output_path, info)
        
        if success:
            results['success'] += 1
        else:
            results['failed'] += 1
            results['errors'].append({
                'file': video_path.name,
                'error': message
            })
    
    return results


def save_processing_report(results: Dict, output_dir: Path):
    """Save processing report to JSON file."""
    report_path = output_dir / "preprocessing_report.json"
    
    # Add summary statistics
    results['summary'] = {
        'target_resolution': f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        'target_fps': TARGET_FPS,
        'target_codec': 'H.264 (libx264)',
        'target_crf': TARGET_CRF,
        'success_rate': f"{results['success']/results['total']*100:.1f}%" if results['total'] > 0 else "N/A"
    }
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nReport saved to: {report_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for video preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess videos for VLM analysis and downstream processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python video_preprocessor.py
    python video_preprocessor.py --input-dir /path/to/videos --output-dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing videos (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for processed videos (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 60)
    print("Video Preprocessor")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS}fps, H.264 CRF {TARGET_CRF}")
    print("=" * 60)
    
    # Process
    results = process_directory(args.input_dir, args.output_dir)
    
    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total:   {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed:  {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    
    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results['errors'][:5]:  # Show first 5
            print(f"  - {err['file']}: {err['error']}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors'])-5} more")
    
    # Save report
    save_processing_report(results, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
