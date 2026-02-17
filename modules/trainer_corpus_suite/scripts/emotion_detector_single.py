#!/usr/bin/env python3
"""
Single-File Emotion Detection with Comprehensive Debugging
==========================================================

Analyzes a single video file for facial emotions using DeepFace.
Produces output identical to emotion_detector.py batch script.

Features:
- Extensive debug logging with timestamps
- Edge-case exception handling with detailed reports
- Processing reports for pipeline integration
- Frame-level issue tracking for debugging

Outputs:
- 7 basic emotions: angry, disgust, fear, happy, sad, surprise, neutral
- Derived Valence/Arousal estimates from emotion distributions
- Detailed processing report with issue diagnostics

Usage:
    python emotion_detector_single.py <video_file> [--output FILE] [--sample-rate N]
    python emotion_detector_single.py <video_file> [--output FILE] [--fps N]
    python emotion_detector_single.py <video_file> --debug --report-dir ./reports
"""

import json
import sys
import warnings
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import numpy as np

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# CONFIGURATION
# =============================================================================

EMOTION_SAMPLE_RATE = 10  # Every 10th frame

# Valence mapping for emotions (positive = pleasure, negative = pain)
EMOTION_VALENCE = {
    'angry': -0.6,
    'disgust': -0.7,
    'fear': -0.8,
    'happy': 0.9,
    'sad': -0.7,
    'surprise': 0.3,
    'neutral': 0.0
}

# Arousal mapping (activation level)
EMOTION_AROUSAL = {
    'angry': 0.8,
    'disgust': 0.5,
    'fear': 0.9,
    'happy': 0.7,
    'sad': 0.3,
    'surprise': 0.9,
    'neutral': 0.2
}

# Processing limits
MAX_CONSECUTIVE_FAILURES = 50  # Stop if this many frames fail in a row
FRAME_TIMEOUT_SECONDS = 30  # Max time per frame analysis
MAX_MEMORY_MB = 4096  # Warn if memory usage exceeds this


# =============================================================================
# LOGGING SETUP
# =============================================================================

class DebugLogger:
    """Centralized logging with timestamps and levels."""
    
    def __init__(self, name: str, debug: bool = False, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.handlers.clear()
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console.setFormatter(fmt)
        self.logger.addHandler(console)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s'
            ))
            self.logger.addHandler(file_handler)
        
        self.debug_mode = debug
    
    def debug(self, msg: str): self.logger.debug(msg)
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def critical(self, msg: str): self.logger.critical(msg)


# =============================================================================
# ISSUE TRACKING
# =============================================================================

class IssueType(Enum):
    """Categories of processing issues."""
    FRAME_READ_ERROR = "frame_read_error"
    FRAME_DECODE_ERROR = "frame_decode_error"
    FRAME_CORRUPT = "frame_corrupt"
    NO_FACE_DETECTED = "no_face_detected"
    FACE_DETECTION_FAILED = "face_detection_failed"
    EMOTION_ANALYSIS_FAILED = "emotion_analysis_failed"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    CODEC_ERROR = "codec_error"
    DIMENSION_ERROR = "dimension_error"
    COLOR_SPACE_ERROR = "color_space_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FrameIssue:
    """Detailed record of a frame processing issue."""
    frame_idx: int
    timestamp_sec: float
    issue_type: str
    error_message: str
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ProcessingReport:
    """Comprehensive processing report for a video."""
    video_path: str
    video_name: str
    processing_started: str
    processing_ended: str
    processing_duration_sec: float
    
    # Video metadata
    total_frames: int
    video_fps: float
    video_duration_sec: float
    video_resolution: Tuple[int, int]
    video_codec: str
    
    # Sampling config
    sample_rate: int
    effective_fps: float
    frames_to_analyze: int
    
    # Results summary
    frames_analyzed: int
    frames_successful: int
    frames_with_faces: int
    frames_failed: int
    
    # Issue tracking
    issues: List[Dict]
    issue_summary: Dict[str, int]
    
    # Consecutive failure tracking
    max_consecutive_failures: int
    failure_ranges: List[Dict]  # [{"start": 100, "end": 150, "count": 50}]
    
    # Performance metrics
    avg_frame_processing_time_ms: float
    max_frame_processing_time_ms: float
    total_processing_time_sec: float
    
    # Recommendations
    recommendations: List[str]
    reprocessing_suggested: bool
    suggested_parameters: Dict[str, Any]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FrameEmotions:
    """Emotion analysis for a single frame."""
    frame_idx: int
    timestamp_sec: float
    num_faces: int
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    valence: float
    arousal: float
    processing_time_ms: float


# =============================================================================
# EDGE CASE HANDLERS
# =============================================================================

class EdgeCaseHandler:
    """Handles various edge cases in frame processing."""
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
        self.issues: List[FrameIssue] = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = 0
        self.failure_ranges = []
        self.current_failure_start = None
    
    def record_issue(self, frame_idx: int, timestamp_sec: float, 
                     issue_type: IssueType, error_msg: str,
                     exception: Optional[Exception] = None) -> FrameIssue:
        """Record a processing issue with full context."""
        
        stack_trace = None
        if exception:
            stack_trace = traceback.format_exc()
        
        # Generate recommendations based on issue type
        recommendations = self._get_recommendations(issue_type, error_msg)
        
        issue = FrameIssue(
            frame_idx=frame_idx,
            timestamp_sec=round(timestamp_sec, 3),
            issue_type=issue_type.value,
            error_message=error_msg,
            stack_trace=stack_trace,
            recommendations=recommendations
        )
        
        self.issues.append(issue)
        self._track_consecutive_failures(frame_idx, failed=True)
        
        self.logger.debug(
            f"Issue @ frame {frame_idx} ({timestamp_sec:.2f}s): "
            f"{issue_type.value} - {error_msg[:100]}"
        )
        
        return issue
    
    def record_success(self, frame_idx: int):
        """Record successful processing (resets consecutive failure counter)."""
        self._track_consecutive_failures(frame_idx, failed=False)
    
    def _track_consecutive_failures(self, frame_idx: int, failed: bool):
        """Track consecutive failure runs for pattern detection."""
        if failed:
            self.consecutive_failures += 1
            if self.current_failure_start is None:
                self.current_failure_start = frame_idx
            self.max_consecutive_failures = max(
                self.max_consecutive_failures, 
                self.consecutive_failures
            )
        else:
            if self.consecutive_failures > 0 and self.current_failure_start is not None:
                self.failure_ranges.append({
                    "start_frame": self.current_failure_start,
                    "end_frame": frame_idx - 1,
                    "count": self.consecutive_failures,
                    "duration_frames": (frame_idx - 1) - self.current_failure_start + 1
                })
            self.consecutive_failures = 0
            self.current_failure_start = None
    
    def finalize(self, last_frame_idx: int):
        """Finalize tracking (close any open failure range)."""
        if self.consecutive_failures > 0 and self.current_failure_start is not None:
            self.failure_ranges.append({
                "start_frame": self.current_failure_start,
                "end_frame": last_frame_idx,
                "count": self.consecutive_failures,
                "duration_frames": last_frame_idx - self.current_failure_start + 1
            })
    
    def _get_recommendations(self, issue_type: IssueType, error_msg: str) -> List[str]:
        """Generate actionable recommendations based on issue type."""
        recs = []
        
        if issue_type == IssueType.FRAME_READ_ERROR:
            recs.append("Video file may be corrupt - try re-encoding with ffmpeg")
            recs.append("Check if video has variable frame rate (VFR) - convert to CFR")
            
        elif issue_type == IssueType.FRAME_DECODE_ERROR:
            recs.append("Codec issue - try: ffmpeg -i input.mp4 -c:v libx264 -crf 18 output.mp4")
            recs.append("Frame may have keyframe dependency issues")
            
        elif issue_type == IssueType.FRAME_CORRUPT:
            recs.append("Frame data is corrupted - extract specific timestamp range")
            recs.append("Consider splitting video at corruption point")
            
        elif issue_type == IssueType.NO_FACE_DETECTED:
            recs.append("No face visible in frame - may be intentional (back of head, etc)")
            recs.append("Try lowering face detection threshold if faces are partially visible")
            
        elif issue_type == IssueType.FACE_DETECTION_FAILED:
            recs.append("Face detector failed - frame may have unusual lighting/angle")
            recs.append("Try preprocessing with histogram equalization")
            
        elif issue_type == IssueType.EMOTION_ANALYSIS_FAILED:
            recs.append("Emotion model failed - face may be too small or occluded")
            recs.append("Ensure face is at least 48x48 pixels")
            
        elif issue_type == IssueType.TIMEOUT:
            recs.append("Processing took too long - reduce resolution or use GPU")
            recs.append("Frame may have many faces - consider face limit parameter")
            
        elif issue_type == IssueType.MEMORY_ERROR:
            recs.append("Out of memory - reduce batch size or frame resolution")
            recs.append("Close other applications or use swap space")
            
        elif issue_type == IssueType.CODEC_ERROR:
            recs.append("Unsupported codec - convert to H.264: ffmpeg -i input -c:v libx264 output.mp4")
            
        elif issue_type == IssueType.DIMENSION_ERROR:
            recs.append("Frame dimensions invalid - check video integrity")
            recs.append(f"Error details: {error_msg}")
            
        elif issue_type == IssueType.COLOR_SPACE_ERROR:
            recs.append("Color space issue - ensure video is RGB/BGR")
            recs.append("Convert colorspace: ffmpeg -i input -pix_fmt yuv420p output.mp4")
        
        return recs
    
    def get_issue_summary(self) -> Dict[str, int]:
        """Get count of each issue type."""
        summary = {}
        for issue in self.issues:
            summary[issue.issue_type] = summary.get(issue.issue_type, 0) + 1
        return summary
    
    def get_global_recommendations(self) -> List[str]:
        """Generate overall recommendations based on all issues."""
        recs = []
        summary = self.get_issue_summary()
        total_issues = len(self.issues)
        
        if total_issues == 0:
            return ["No issues detected - processing completed successfully"]
        
        # High failure rate
        if self.max_consecutive_failures > 20:
            recs.append(
                f"CRITICAL: {self.max_consecutive_failures} consecutive failures detected. "
                f"Video may have corrupt section. Check failure ranges in report."
            )
        
        # Codec issues
        codec_issues = summary.get(IssueType.CODEC_ERROR.value, 0)
        if codec_issues > 0:
            recs.append(
                "Re-encode video with standard codec: "
                "ffmpeg -i input.mp4 -c:v libx264 -preset medium -crf 18 -c:a aac output.mp4"
            )
        
        # Face detection issues
        no_face = summary.get(IssueType.NO_FACE_DETECTED.value, 0)
        face_failed = summary.get(IssueType.FACE_DETECTION_FAILED.value, 0)
        if no_face > total_issues * 0.5:
            recs.append(
                "Over 50% of frames had no faces detected. "
                "This may be normal for this content, or face detection threshold may need adjustment."
            )
        
        # Memory issues
        mem_issues = summary.get(IssueType.MEMORY_ERROR.value, 0)
        if mem_issues > 0:
            recs.append(
                "Memory errors detected. Consider: "
                "1) Closing other applications, "
                "2) Processing at lower resolution, "
                "3) Increasing system swap space"
            )
        
        # Timeout issues
        timeout_issues = summary.get(IssueType.TIMEOUT.value, 0)
        if timeout_issues > 0:
            recs.append(
                f"{timeout_issues} frames timed out. Consider using GPU acceleration "
                "or reducing the number of faces analyzed per frame."
            )
        
        return recs


# =============================================================================
# EMOTION ANALYSIS
# =============================================================================

class EmotionAnalyzer:
    """DeepFace-based emotion analyzer with comprehensive error handling."""
    
    def __init__(self, logger: DebugLogger):
        self.deepface = None
        self.logger = logger
        self.edge_handler = EdgeCaseHandler(logger)
        self._load_model()
    
    def _load_model(self):
        try:
            self.logger.debug("Loading DeepFace library...")
            start = time.time()
            from deepface import DeepFace
            self.deepface = DeepFace
            load_time = time.time() - start
            self.logger.info(f"DeepFace loaded successfully ({load_time:.2f}s)")
        except ImportError as e:
            self.logger.critical(f"DeepFace not available: {e}")
            self.logger.critical("Install with: pip install deepface tensorflow")
            sys.exit(1)
        except Exception as e:
            self.logger.critical(f"Failed to load DeepFace: {e}")
            sys.exit(1)
    
    def analyze_frame(self, frame: np.ndarray, frame_idx: int, 
                      timestamp_sec: float) -> Optional[FrameEmotions]:
        """Analyze emotions in a single frame with comprehensive error handling."""
        
        if self.deepface is None:
            return None
        
        start_time = time.time()
        
        # Validate frame
        issue = self._validate_frame(frame, frame_idx, timestamp_sec)
        if issue:
            return None
        
        try:
            self.logger.debug(f"Analyzing frame {frame_idx} ({timestamp_sec:.2f}s)...")
            
            # Run DeepFace analysis
            results = self.deepface.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if not results:
                self.edge_handler.record_issue(
                    frame_idx, timestamp_sec,
                    IssueType.NO_FACE_DETECTED,
                    "DeepFace returned empty results"
                )
                return None
            
            if isinstance(results, dict):
                results = [results]
            
            # Extract emotions from all faces
            all_emotions = []
            for i, face_result in enumerate(results):
                if 'emotion' not in face_result:
                    self.logger.debug(f"  Face {i}: no emotion data")
                    continue
                    
                # Check face confidence if available
                face_conf = face_result.get('face_confidence', 1.0)
                if face_conf < 0.5:
                    self.logger.debug(f"  Face {i}: low confidence ({face_conf:.2f}), skipping")
                    continue
                    
                all_emotions.append(face_result['emotion'])
                self.logger.debug(
                    f"  Face {i}: dominant={max(face_result['emotion'], key=face_result['emotion'].get)}"
                )
            
            if not all_emotions:
                self.edge_handler.record_issue(
                    frame_idx, timestamp_sec,
                    IssueType.NO_FACE_DETECTED,
                    "No valid faces with emotion data found"
                )
                return None
            
            # Average emotion scores across all faces
            avg_emotions = {}
            for emotion in EMOTION_VALENCE.keys():
                scores = [e.get(emotion, 0) for e in all_emotions]
                avg_emotions[emotion] = np.mean(scores)
            
            # Normalize to sum to 100
            total = sum(avg_emotions.values())
            if total > 0:
                avg_emotions = {k: v/total * 100 for k, v in avg_emotions.items()}
            
            dominant = max(avg_emotions, key=avg_emotions.get)
            valence = self._calculate_valence(avg_emotions)
            arousal = self._calculate_arousal(avg_emotions)
            
            self.edge_handler.record_success(frame_idx)
            
            self.logger.debug(
                f"  Result: {dominant} (v={valence:.2f}, a={arousal:.2f}) "
                f"[{processing_time:.0f}ms, {len(all_emotions)} face(s)]"
            )
            
            return FrameEmotions(
                frame_idx=frame_idx,
                timestamp_sec=round(timestamp_sec, 3),
                num_faces=len(all_emotions),
                dominant_emotion=dominant,
                emotion_scores={k: round(v, 2) for k, v in avg_emotions.items()},
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                processing_time_ms=round(processing_time, 2)
            )
            
        except MemoryError as e:
            self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                IssueType.MEMORY_ERROR,
                str(e), e
            )
            return None
            
        except Exception as e:
            error_msg = str(e)
            
            # Categorize the error
            if "face" in error_msg.lower() and "detect" in error_msg.lower():
                issue_type = IssueType.FACE_DETECTION_FAILED
            elif "emotion" in error_msg.lower():
                issue_type = IssueType.EMOTION_ANALYSIS_FAILED
            elif "memory" in error_msg.lower() or "alloc" in error_msg.lower():
                issue_type = IssueType.MEMORY_ERROR
            else:
                issue_type = IssueType.UNKNOWN_ERROR
            
            self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                issue_type, error_msg, e
            )
            return None
    
    def _validate_frame(self, frame: np.ndarray, frame_idx: int, 
                        timestamp_sec: float) -> Optional[FrameIssue]:
        """Validate frame data before processing."""
        
        # Check if frame is None
        if frame is None:
            return self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                IssueType.FRAME_READ_ERROR,
                "Frame is None"
            )
        
        # Check dimensions
        if len(frame.shape) != 3:
            return self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                IssueType.DIMENSION_ERROR,
                f"Invalid frame dimensions: {frame.shape}"
            )
        
        h, w, c = frame.shape
        
        if h < 10 or w < 10:
            return self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                IssueType.DIMENSION_ERROR,
                f"Frame too small: {w}x{h}"
            )
        
        if c not in [1, 3, 4]:
            return self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                IssueType.COLOR_SPACE_ERROR,
                f"Invalid channel count: {c}"
            )
        
        # Check for corrupt data (all zeros or all same value)
        if np.all(frame == frame[0, 0, 0]):
            return self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                IssueType.FRAME_CORRUPT,
                "Frame is uniform (possibly corrupt)"
            )
        
        # Check for NaN or Inf values
        if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
            return self.edge_handler.record_issue(
                frame_idx, timestamp_sec,
                IssueType.FRAME_CORRUPT,
                "Frame contains NaN or Inf values"
            )
        
        return None
    
    def _calculate_valence(self, emotions: Dict[str, float]) -> float:
        valence = 0.0
        for emotion, score in emotions.items():
            weight = score / 100
            valence += weight * EMOTION_VALENCE.get(emotion, 0)
        return valence
    
    def _calculate_arousal(self, emotions: Dict[str, float]) -> float:
        arousal = 0.0
        for emotion, score in emotions.items():
            weight = score / 100
            arousal += weight * EMOTION_AROUSAL.get(emotion, 0)
        return arousal
    
    def analyze_video(self, video_path: Path, sample_rate: int = None, 
                      target_fps: float = None) -> Tuple[Dict, ProcessingReport]:
        """
        Analyze emotions throughout a video with full diagnostic reporting.
        
        Returns:
            Tuple of (analysis_results, processing_report)
        """
        import cv2
        
        processing_started = datetime.now().isoformat()
        start_time = time.time()
        
        self.logger.info(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            self.logger.error(f"Could not open video: {video_path}")
            report = self._create_error_report(video_path, "Failed to open video file")
            return self._empty_analysis(video_path), report
        
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        duration_sec = total_frames / video_fps if video_fps > 0 else 0
        
        self.logger.info(f"Video metadata:")
        self.logger.info(f"  Resolution: {width}x{height}")
        self.logger.info(f"  FPS: {video_fps:.2f}")
        self.logger.info(f"  Total frames: {total_frames}")
        self.logger.info(f"  Duration: {duration_sec:.2f}s")
        self.logger.info(f"  Codec: {codec}")
        
        # Determine sample rate
        if target_fps is not None:
            sample_rate = max(1, int(round(video_fps / target_fps)))
            effective_fps = video_fps / sample_rate
            self.logger.info(f"Target FPS: {target_fps} -> sample_rate={sample_rate} ({effective_fps:.2f} effective FPS)")
        else:
            sample_rate = sample_rate or EMOTION_SAMPLE_RATE
            effective_fps = video_fps / sample_rate
            self.logger.info(f"Sample rate: every {sample_rate} frames ({effective_fps:.2f} effective FPS)")
        
        frames_to_analyze = total_frames // sample_rate
        self.logger.info(f"Frames to analyze: ~{frames_to_analyze}")
        
        # Process frames
        frame_emotions = []
        frame_processing_times = []
        frame_idx = 0
        frames_analyzed = 0
        
        self.logger.info("Starting frame analysis...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                if frame_idx < total_frames - 1:
                    # Premature end - record issue
                    timestamp = frame_idx / video_fps if video_fps > 0 else 0
                    self.edge_handler.record_issue(
                        frame_idx, timestamp,
                        IssueType.FRAME_READ_ERROR,
                        f"Video ended prematurely at frame {frame_idx}/{total_frames}"
                    )
                    self.logger.warning(f"Video ended at frame {frame_idx}, expected {total_frames}")
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                frames_analyzed += 1
                timestamp_sec = frame_idx / video_fps if video_fps > 0 else 0
                
                # Check consecutive failures
                if self.edge_handler.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.logger.error(
                        f"STOPPING: {MAX_CONSECUTIVE_FAILURES} consecutive failures. "
                        f"Video may be corrupt starting at frame {self.edge_handler.current_failure_start}"
                    )
                    break
                
                emotions = self.analyze_frame(frame, frame_idx, timestamp_sec)
                if emotions:
                    frame_emotions.append(emotions)
                    frame_processing_times.append(emotions.processing_time_ms)
                
                # Progress logging (every 10%)
                progress = (frame_idx / total_frames) * 100
                if frames_analyzed % max(1, frames_to_analyze // 10) == 0:
                    self.logger.info(
                        f"Progress: {progress:.1f}% ({frames_analyzed}/{frames_to_analyze} frames, "
                        f"{len(frame_emotions)} successful)"
                    )
            
            frame_idx += 1
        
        cap.release()
        
        # Finalize edge case tracking
        self.edge_handler.finalize(frame_idx)
        
        processing_ended = datetime.now().isoformat()
        processing_duration = time.time() - start_time
        
        self.logger.info(f"Processing complete in {processing_duration:.2f}s")
        self.logger.info(f"Frames analyzed: {frames_analyzed}")
        self.logger.info(f"Frames successful: {len(frame_emotions)}")
        self.logger.info(f"Issues encountered: {len(self.edge_handler.issues)}")
        
        # Create processing report
        report = ProcessingReport(
            video_path=str(video_path),
            video_name=video_path.name,
            processing_started=processing_started,
            processing_ended=processing_ended,
            processing_duration_sec=round(processing_duration, 2),
            total_frames=total_frames,
            video_fps=round(video_fps, 2),
            video_duration_sec=round(duration_sec, 2),
            video_resolution=(width, height),
            video_codec=codec,
            sample_rate=sample_rate,
            effective_fps=round(effective_fps, 2),
            frames_to_analyze=frames_to_analyze,
            frames_analyzed=frames_analyzed,
            frames_successful=len(frame_emotions),
            frames_with_faces=sum(1 for e in frame_emotions if e.num_faces > 0),
            frames_failed=frames_analyzed - len(frame_emotions),
            issues=[asdict(i) for i in self.edge_handler.issues],
            issue_summary=self.edge_handler.get_issue_summary(),
            max_consecutive_failures=self.edge_handler.max_consecutive_failures,
            failure_ranges=self.edge_handler.failure_ranges,
            avg_frame_processing_time_ms=round(np.mean(frame_processing_times), 2) if frame_processing_times else 0,
            max_frame_processing_time_ms=round(max(frame_processing_times), 2) if frame_processing_times else 0,
            total_processing_time_sec=round(processing_duration, 2),
            recommendations=self.edge_handler.get_global_recommendations(),
            reprocessing_suggested=self.edge_handler.max_consecutive_failures > 10,
            suggested_parameters=self._suggest_parameters(frame_emotions, frames_analyzed)
        )
        
        # Generate analysis results
        results = self._aggregate_emotions(video_path, frame_emotions, frames_analyzed)
        
        return results, report
    
    def _suggest_parameters(self, frame_emotions: List[FrameEmotions], 
                           frames_analyzed: int) -> Dict[str, Any]:
        """Suggest optimal parameters for reprocessing."""
        suggestions = {}
        
        success_rate = len(frame_emotions) / frames_analyzed if frames_analyzed > 0 else 0
        
        if success_rate < 0.3:
            suggestions["sample_rate"] = "increase"
            suggestions["note"] = "Low success rate - video may have limited face visibility"
        
        if frame_emotions:
            avg_time = np.mean([e.processing_time_ms for e in frame_emotions])
            if avg_time > 1000:
                suggestions["use_gpu"] = True
                suggestions["note"] = "High processing time - GPU acceleration recommended"
        
        return suggestions
    
    def _create_error_report(self, video_path: Path, error: str) -> ProcessingReport:
        """Create a report for videos that failed to open."""
        now = datetime.now().isoformat()
        return ProcessingReport(
            video_path=str(video_path),
            video_name=video_path.name,
            processing_started=now,
            processing_ended=now,
            processing_duration_sec=0,
            total_frames=0,
            video_fps=0,
            video_duration_sec=0,
            video_resolution=(0, 0),
            video_codec="unknown",
            sample_rate=0,
            effective_fps=0,
            frames_to_analyze=0,
            frames_analyzed=0,
            frames_successful=0,
            frames_with_faces=0,
            frames_failed=0,
            issues=[{
                "frame_idx": -1,
                "timestamp_sec": 0,
                "issue_type": "video_open_failed",
                "error_message": error,
                "recommendations": [
                    "Check if file exists and is readable",
                    "Verify video file is not corrupt",
                    "Try opening with ffprobe to check file integrity"
                ]
            }],
            issue_summary={"video_open_failed": 1},
            max_consecutive_failures=0,
            failure_ranges=[],
            avg_frame_processing_time_ms=0,
            max_frame_processing_time_ms=0,
            total_processing_time_sec=0,
            recommendations=["Video could not be opened - check file integrity"],
            reprocessing_suggested=True,
            suggested_parameters={"re-encode": True}
        )
    
    def _empty_analysis(self, video_path: Path) -> Dict:
        return {
            'scene_path': str(video_path),
            'total_frames_analyzed': 0,
            'frames_with_faces': 0,
            'dominant_emotion': 'neutral',
            'emotion_distribution': {e: 0 for e in EMOTION_VALENCE.keys()},
            'mean_valence': 0,
            'mean_arousal': 0,
            'valence_range': [0, 0],
            'arousal_range': [0, 0],
            'pain_pleasure_score': 0,
            'frame_emotions': []
        }
    
    def _aggregate_emotions(self, video_path: Path, 
                           frame_emotions: List[FrameEmotions],
                           total_analyzed: int) -> Dict:
        """Aggregate frame-level emotions into scene-level analysis."""
        
        if not frame_emotions:
            return self._empty_analysis(video_path)
        
        avg_emotions = {e: 0 for e in EMOTION_VALENCE.keys()}
        for fe in frame_emotions:
            for emotion, score in fe.emotion_scores.items():
                avg_emotions[emotion] += score
        
        n = len(frame_emotions)
        avg_emotions = {k: round(v/n, 2) for k, v in avg_emotions.items()}
        
        dominant = max(avg_emotions, key=avg_emotions.get)
        
        valences = [fe.valence for fe in frame_emotions]
        arousals = [fe.arousal for fe in frame_emotions]
        
        mean_valence = np.mean(valences)
        mean_arousal = np.mean(arousals)
        
        pain_pleasure = mean_valence * (0.5 + 0.5 * mean_arousal)
        
        return {
            'scene_path': str(video_path),
            'total_frames_analyzed': total_analyzed,
            'frames_with_faces': n,
            'dominant_emotion': dominant,
            'emotion_distribution': avg_emotions,
            'mean_valence': round(mean_valence, 3),
            'mean_arousal': round(mean_arousal, 3),
            'valence_range': [round(min(valences), 3), round(max(valences), 3)],
            'arousal_range': [round(min(arousals), 3), round(max(arousals), 3)],
            'pain_pleasure_score': round(pain_pleasure, 3),
            'frame_emotions': [asdict(fe) for fe in frame_emotions]
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze emotions in a single video file with comprehensive debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                          # Basic analysis
  %(prog)s video.mp4 --fps 2                  # 2 frames per second
  %(prog)s video.mp4 --debug --report-dir .   # Full debug with reports
  %(prog)s video.mp4 -o result.json -r 5      # Custom sample rate
        """
    )
    
    parser.add_argument(
        'video',
        type=Path,
        help="Path to video file"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help="Output JSON file for analysis results"
    )
    
    parser.add_argument(
        '--report-dir',
        type=Path,
        default=None,
        help="Directory to save processing report (creates <video>_report.json)"
    )
    
    parser.add_argument(
        '--log-file',
        type=Path,
        default=None,
        help="File to save debug log"
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help="Enable verbose debug output"
    )
    
    # Mutually exclusive: sample-rate vs fps
    sampling_group = parser.add_mutually_exclusive_group()
    
    sampling_group.add_argument(
        '--sample-rate', '-r',
        type=int,
        default=None,
        help=f"Process every Nth frame (default: {EMOTION_SAMPLE_RATE})"
    )
    
    sampling_group.add_argument(
        '--fps', '-f',
        type=float,
        default=None,
        help="Target frames per second to analyze"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file
    if args.debug and not log_file and args.report_dir:
        log_file = args.report_dir / f"{args.video.stem}_debug.log"
    
    logger = DebugLogger("emotion_detector", debug=args.debug, log_file=log_file)
    
    if not args.video.exists():
        logger.critical(f"Video file not found: {args.video}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("EMOTION DETECTOR - Single File Analysis")
    logger.info("=" * 70)
    logger.info(f"Video: {args.video}")
    logger.info(f"Debug mode: {args.debug}")
    if args.fps:
        logger.info(f"Target FPS: {args.fps}")
    elif args.sample_rate:
        logger.info(f"Sample rate: {args.sample_rate}")
    else:
        logger.info(f"Sample rate: {EMOTION_SAMPLE_RATE} (default)")
    logger.info("=" * 70)
    
    # Run analysis
    analyzer = EmotionAnalyzer(logger)
    results, report = analyzer.analyze_video(
        args.video, 
        sample_rate=args.sample_rate, 
        target_fps=args.fps
    )
    
    # Output results summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Frames analyzed:     {report.frames_analyzed}")
    logger.info(f"Frames successful:   {report.frames_successful}")
    logger.info(f"Frames with faces:   {report.frames_with_faces}")
    logger.info(f"Frames failed:       {report.frames_failed}")
    logger.info(f"Dominant emotion:    {results['dominant_emotion']}")
    logger.info(f"Mean valence:        {results['mean_valence']:.3f}")
    logger.info(f"Mean arousal:        {results['mean_arousal']:.3f}")
    logger.info(f"Pain/Pleasure score: {results['pain_pleasure_score']:.3f}")
    logger.info(f"Processing time:     {report.processing_duration_sec:.2f}s")
    
    logger.info("")
    logger.info("Emotion Distribution:")
    for emotion, score in sorted(results['emotion_distribution'].items(), key=lambda x: -x[1]):
        logger.info(f"  {emotion:10s}: {score:5.1f}%")
    
    # Issue summary
    if report.issues:
        logger.info("")
        logger.info("=" * 70)
        logger.info("ISSUE SUMMARY")
        logger.info("=" * 70)
        for issue_type, count in sorted(report.issue_summary.items(), key=lambda x: -x[1]):
            logger.info(f"  {issue_type}: {count}")
        
        if report.failure_ranges:
            logger.info("")
            logger.info("Failure Ranges (consecutive failures):")
            for fr in report.failure_ranges:
                logger.info(
                    f"  Frames {fr['start_frame']}-{fr['end_frame']} "
                    f"({fr['count']} failures)"
                )
    
    # Recommendations
    if report.recommendations:
        logger.info("")
        logger.info("=" * 70)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 70)
        for i, rec in enumerate(report.recommendations, 1):
            logger.info(f"  {i}. {rec}")
    
    # Save outputs
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nAnalysis saved to: {args.output}")
    
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.report_dir / f"{args.video.stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Processing report saved to: {report_path}")
        
        # Also save config used
        config_path = args.report_dir / f"{args.video.stem}_config.json"
        config = {
            "video": str(args.video),
            "sample_rate": args.sample_rate or EMOTION_SAMPLE_RATE,
            "target_fps": args.fps,
            "debug": args.debug,
            "emotion_valence_mapping": EMOTION_VALENCE,
            "emotion_arousal_mapping": EMOTION_AROUSAL,
            "max_consecutive_failures": MAX_CONSECUTIVE_FAILURES,
            "timestamp": datetime.now().isoformat()
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to: {config_path}")
    
    # Print JSON to stdout if no output file specified
    if not args.output and not args.report_dir:
        logger.info("")
        logger.info("-" * 70)
        logger.info("JSON Output (use -o to save to file):")
        print(json.dumps(results, indent=2))
    
    logger.info("")
    logger.info("Done!")
    
    # Exit with error code if there were critical issues
    if report.reprocessing_suggested:
        logger.warning("Reprocessing suggested due to high failure rate")
        sys.exit(2)


if __name__ == "__main__":
    main()
