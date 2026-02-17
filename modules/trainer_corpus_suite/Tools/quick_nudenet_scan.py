#!/usr/bin/env python3
"""
Quick NudeNet Video Scanner
===========================

One-off script to quickly scan videos in ./processed for nudity content.
No data saving - just prints results to console.

Usage:
    conda activate lora_trainer_suite
    python Tools/quick_nudenet_scan.py

Or run specific videos:
    python Tools/quick_nudenet_scan.py --videos video1.mp4 video2.mp4
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# NudeNet for detection
try:
    import nudenet as nn
except ImportError:
    print("ERROR: NudeNet not found. Please install with: pip install nudenet")
    print("Or run in lora_trainer_suite environment: source Lora/lora_trainer_suite/venv/bin/activate")
    sys.exit(1)


# Configuration
PROCESSED_DIR = Path(__file__).parent.parent / "processed"
FRAME_SAMPLE_RATE = 60  # Process every Nth frame (more aggressive sampling for quick scan)
NUDENET_MODEL_SIZE = 640
VIDEO_EXTENSIONS = {'.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV'}


class QuickNudeNetScanner:
    """Quick scanner for nudity detection without data persistence."""

    def __init__(self, model_size: int = NUDENET_MODEL_SIZE):
        """Initialize NudeNet model."""
        print("Loading NudeNet model...")
        from nudenet import NudeDetector
        self.model = NudeDetector()
        print("✓ Model loaded")

    def detect_frame(self, frame: np.ndarray) -> List[dict]:
        """Run NudeNet detection on a single frame."""
        # Convert BGR to RGB for NudeNet
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        detections = self.model.detect(rgb_frame)

        # Filter for exposed content only (most relevant for quick scan)
        exposed_labels = {
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
        }

        filtered = []
        for detection in detections:
            # NudeNet returns 'class' not 'label'
            label = detection.get('class', '')
            if label in exposed_labels:
                filtered.append({
                    'label': label,
                    'score': detection.get('score', 0.0),
                    'box': detection.get('box', []),
                })

        return filtered

    def scan_video(self, video_path: Path, sample_rate: int = FRAME_SAMPLE_RATE) -> dict:
        """Quick scan of a video for nudity content."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {'error': f"Could not open video: {video_path.name}"}

        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        detections_found = []
        frames_with_nudity = 0
        total_processed = 0

        print(f"  Scanning {video_path.name} ({total_frames} frames, {fps:.1f} fps)")

        # Quick scan with progress bar
        with tqdm(total=min(total_frames // sample_rate, 100), desc="    Scanning", leave=False) as pbar:
            frame_idx = 0
            scan_count = 0

            while scan_count < 100:  # Limit to 100 samples for quick scan
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames
                if frame_idx % sample_rate == 0:
                    detections = self.detect_frame(frame)

                    if detections:
                        frames_with_nudity += 1
                        detections_found.extend(detections)

                        # Show immediate feedback for first detection
                        if len(detections_found) <= 3:  # Show first few detections
                            labels = [d['label'] for d in detections]
                            max_score = max(d['score'] for d in detections)
                            print(f"    ⚠️  Frame {frame_idx}: {labels} (conf: {max_score:.2f})")

                    total_processed += 1
                    scan_count += 1
                    pbar.update(1)

                frame_idx += 1

        cap.release()

        # Calculate results
        nudity_rate = frames_with_nudity / max(total_processed, 1)

        result = {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'total_frames': total_frames,
            'frames_processed': total_processed,
            'frames_with_nudity': frames_with_nudity,
            'nudity_rate': nudity_rate,
            'total_detections': len(detections_found),
            'has_nudity': frames_with_nudity > 0,
        }

        # Add detection summary
        if detections_found:
            label_counts = {}
            for detection in detections_found:
                label = detection['label']
                label_counts[label] = label_counts.get(label, 0) + 1

            result['detection_summary'] = label_counts
            result['avg_confidence'] = np.mean([d['score'] for d in detections_found])

        return result

    def scan_multiple_videos(self, video_paths: List[Path]) -> List[dict]:
        """Scan multiple videos and return results."""
        results = []

        for video_path in video_paths:
            result = self.scan_video(video_path)
            results.append(result)

            # Print summary for this video
            if 'error' in result:
                print(f"❌ {result['error']}")
            else:
                nudity_indicator = "🔴 NUDITY DETECTED" if result['has_nudity'] else "✅ Clean"
                print(f"📹 {result['video_name']}: {nudity_indicator}")
                if result['has_nudity']:
                    print(".2%")
                    if 'detection_summary' in result:
                        for label, count in result['detection_summary'].items():
                            print(f"    {label}: {count} detections")

        return results


def discover_videos(input_dir: Path) -> List[Path]:
    """Find all video files in input directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_dir.glob(f"*{ext}"))
    return sorted(videos)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick NudeNet video scanner - no data saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Tools/quick_nudenet_scan.py                    # Scan all videos in ./processed
  python Tools/quick_nudenet_scan.py --videos vid1.mp4 vid2.mp4  # Scan specific videos
  python Tools/quick_nudenet_scan.py --dir /path/to/videos       # Scan different directory
        """
    )

    parser.add_argument(
        '--videos', '-v', nargs='*',
        help="Specific video files to scan"
    )

    parser.add_argument(
        '--dir', '-d', type=Path, default=PROCESSED_DIR,
        help=f"Directory to scan (default: {PROCESSED_DIR})"
    )

    parser.add_argument(
        '--sample-rate', '-s', type=int, default=FRAME_SAMPLE_RATE,
        help=f"Process every Nth frame (default: {FRAME_SAMPLE_RATE})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Quick NudeNet Video Scanner")
    print("=" * 60)
    print(f"Sample Rate: Every {args.sample_rate} frames")
    print(f"Model Size: {NUDENET_MODEL_SIZE}px")
    print("=" * 60)

    # Determine which videos to scan
    if args.videos:
        # Specific videos provided
        video_paths = [Path(v) for v in args.videos]
        print(f"Scanning {len(video_paths)} specified videos...")
    else:
        # Scan directory
        if not args.dir.exists():
            print(f"Error: Directory does not exist: {args.dir}")
            sys.exit(1)

        video_paths = discover_videos(args.dir)
        print(f"Found {len(video_paths)} videos in {args.dir}")

    if not video_paths:
        print("No videos to scan!")
        sys.exit(1)

    # Initialize scanner
    scanner = QuickNudeNetScanner()

    # Scan videos
    results = scanner.scan_multiple_videos(video_paths)

    # Print overall summary
    print("\n" + "=" * 60)
    print("SCAN SUMMARY")
    print("=" * 60)

    total_videos = len(results)
    videos_with_nudity = sum(1 for r in results if r.get('has_nudity', False))
    total_detections = sum(r.get('total_detections', 0) for r in results)

    print(f"Total Videos Scanned: {total_videos}")
    print(f"Videos with Nudity: {videos_with_nudity}")
    print(f"Videos Clean: {total_videos - videos_with_nudity}")
    print(f"Total Detections: {total_detections}")

    if videos_with_nudity > 0:
        print(f"\n⚠️  Nudity detected in {videos_with_nudity} video(s)")
        print("Check individual video results above for details")
    else:
        print("\n✅ All scanned videos appear clean")

    print("=" * 60)


if __name__ == "__main__":
    main()