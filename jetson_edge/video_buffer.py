#!/usr/bin/env python3
"""
Video Ring Buffer for Event Clip Capture.

Implements a 15-second rolling buffer that continuously stores video frames
on SSD. When an event is triggered, extracts pre-event and post-event footage
into a compressed clip.

Architecture:
    - Ring buffer stores raw frames in memory (fast write)
    - On event: extract frames, encode with H.265 (NVENC), save to disk
    - Buffer runs on separate thread to avoid blocking main pipeline

Usage:
    buffer = VideoRingBuffer(
        buffer_seconds=15,
        fps=25,
        buffer_path="/opt/qraie/data/video_buffer"
    )
    buffer.start()
    
    # In main loop:
    buffer.add_frame(frame)
    
    # On event (event_id should be device_id + event_id):
    clip_event_id = f"{device_id}_{event_id}"
    clip_path = buffer.capture_event_clip(clip_event_id)
    # Output: /opt/qraie/data/video_buffer/cam-001_EV1-1234567890-abc12345.mp4
    
    # Cleanup:
    buffer.stop()
"""

import logging
import os
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Deque, Tuple, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for a single frame with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_id: int


class VideoRingBuffer:
    """
    Ring buffer for continuous video capture with event-triggered clip extraction.
    
    Features:
    - Circular buffer holding N seconds of frames
    - Thread-safe frame addition
    - Event-triggered clip extraction with pre/post event footage
    - H.265 encoding with hardware acceleration (NVENC on Jetson)
    - Automatic cleanup of old clips
    """
    
    def __init__(
        self,
        buffer_seconds: int = 15,
        fps: int = 25,
        buffer_path: str = "/opt/qraie/data/video_buffer",
        pre_event_seconds: int = 10,
        post_event_seconds: int = 5,
        resolution: Tuple[int, int] = (1280, 720),
        codec: str = "h265",
        crf: int = 28,
        max_clips: int = 100,
        max_storage_mb: int = 1024,
    ):
        """
        Initialize video ring buffer.
        
        Args:
            buffer_seconds: Total buffer duration in seconds
            fps: Expected frame rate
            buffer_path: Directory for saving clips
            pre_event_seconds: Seconds of footage before event
            post_event_seconds: Seconds of footage after event
            resolution: Target resolution (width, height)
            codec: Video codec (h265 or h264)
            crf: Constant Rate Factor for quality (lower = better)
            max_clips: Maximum clips to retain
            max_storage_mb: Maximum storage for clips in MB
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.buffer_path = Path(buffer_path)
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.resolution = resolution
        self.codec = codec
        self.crf = crf
        self.max_clips = max_clips
        self.max_storage_mb = max_storage_mb
        
        # Calculate buffer size
        self.max_frames = buffer_seconds * fps
        
        # Ring buffer (thread-safe deque)
        self._buffer: Deque[FrameData] = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
        
        # State
        self._running = False
        self._frame_count = 0
        self._pending_captures: list = []  # (event_id, trigger_time, frames_remaining)
        self._capture_thread: Optional[threading.Thread] = None
        
        # Callback when clip is ready: fn(event_id: str, clip_path: str) -> None
        self.on_clip_ready: Optional[Callable[[str, str], None]] = None
        
        # Statistics
        self.stats = {
            "frames_added": 0,
            "clips_captured": 0,
            "clips_failed": 0,
            "bytes_written": 0
        }
        
        # Ensure buffer directory exists
        self.buffer_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"VideoRingBuffer initialized: {buffer_seconds}s @ {fps}fps, "
            f"max_frames={self.max_frames}, path={buffer_path}"
        )
    
    def start(self) -> None:
        """Start the buffer (enables capture processing thread)."""
        if self._running:
            return
        
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True,
            name="VideoBuffer-Capture"
        )
        self._capture_thread.start()
        logger.info("VideoRingBuffer started")
    
    def stop(self) -> None:
        """Stop the buffer and cleanup."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)
        logger.info(f"VideoRingBuffer stopped. Stats: {self.stats}")
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the ring buffer.
        
        Args:
            frame: BGR image frame (numpy array)
        """
        if not self._running:
            return
        
        timestamp = time.time()
        self._frame_count += 1
        
        # Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution)
        
        frame_data = FrameData(
            frame=frame.copy(),  # Copy to avoid reference issues
            timestamp=timestamp,
            frame_id=self._frame_count
        )
        
        with self._lock:
            self._buffer.append(frame_data)
        
        self.stats["frames_added"] += 1
        
        # Update pending captures (count down post-event frames)
        for i, (event_id, trigger_time, remaining) in enumerate(self._pending_captures):
            if remaining > 0:
                self._pending_captures[i] = (event_id, trigger_time, remaining - 1)
    
    def capture_event_clip(self, event_id: str, delay_frames: int = None) -> Optional[str]:
        """
        Trigger clip capture for an event.
        
        The clip will include pre_event_seconds before and post_event_seconds
        after the trigger point. Post-event frames are collected asynchronously.
        
        Args:
            event_id: Unique event identifier for the clip filename.
                      Should be formatted as {device_id}_{event_id} 
                      (e.g., "cam-001_EV1-1234567890-abc12345")
            delay_frames: Override post-event frame count
            
        Returns:
            Path to clip file (async - may not exist immediately)
        """
        if delay_frames is None:
            delay_frames = self.post_event_seconds * self.fps
        
        trigger_time = time.time()
        
        # Queue capture (will be processed when post-event frames collected)
        self._pending_captures.append((event_id, trigger_time, delay_frames))
        
        # Generate expected clip path (event_id should be device_id + event_id)
        clip_filename = f"{event_id}.mp4"
        clip_path = self.buffer_path / clip_filename
        
        logger.info(f"Queued clip capture: {event_id}, post-frames={delay_frames}")
        
        return str(clip_path)
    
    def _capture_worker(self) -> None:
        """Background worker for processing clip captures."""
        while self._running:
            # Check for completed captures
            completed = []
            
            for i, (event_id, trigger_time, remaining) in enumerate(self._pending_captures):
                if remaining <= 0:
                    completed.append(i)
                    self._process_capture(event_id, trigger_time)
            
            # Remove completed captures (in reverse to maintain indices)
            for i in reversed(completed):
                self._pending_captures.pop(i)
            
            time.sleep(0.1)  # Check every 100ms
    
    def _process_capture(self, event_id: str, trigger_time: float) -> Optional[str]:
        """
        Process a clip capture.
        
        Args:
            event_id: Event identifier
            trigger_time: When the event was triggered
            
        Returns:
            Path to saved clip, or None on failure
        """
        try:
            # Get frames from buffer
            pre_frames = self.pre_event_seconds * self.fps
            post_frames = self.post_event_seconds * self.fps
            
            with self._lock:
                # Find frames around trigger time
                all_frames = list(self._buffer)
            
            if not all_frames:
                logger.warning(f"No frames in buffer for clip {event_id}")
                return None
            
            # Find trigger frame index
            trigger_idx = None
            for i, fd in enumerate(all_frames):
                if fd.timestamp >= trigger_time:
                    trigger_idx = i
                    break
            
            if trigger_idx is None:
                trigger_idx = len(all_frames) - 1
            
            # Extract frame range
            start_idx = max(0, trigger_idx - pre_frames)
            end_idx = min(len(all_frames), trigger_idx + post_frames)
            
            clip_frames = all_frames[start_idx:end_idx]
            
            if len(clip_frames) < self.fps:  # At least 1 second
                logger.warning(f"Insufficient frames for clip {event_id}: {len(clip_frames)}")
                return None
            
            # Generate clip (event_id should be device_id + event_id)
            clip_filename = f"{event_id}.mp4"
            clip_path = self.buffer_path / clip_filename
            
            # Encode frames to video
            success = self._encode_clip(clip_frames, clip_path)
            
            if success:
                self.stats["clips_captured"] += 1
                file_size = clip_path.stat().st_size
                self.stats["bytes_written"] += file_size
                logger.info(f"Clip saved: {clip_path} ({file_size/1024:.1f} KB, {len(clip_frames)} frames)")
                
                # Cleanup old clips if needed
                self._cleanup_old_clips()
                
                # Invoke callback if registered
                if self.on_clip_ready:
                    try:
                        self.on_clip_ready(event_id, str(clip_path))
                    except Exception as cb_err:
                        logger.error(f"Clip ready callback failed: {cb_err}")
                
                return str(clip_path)
            else:
                self.stats["clips_failed"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Clip capture failed for {event_id}: {e}")
            self.stats["clips_failed"] += 1
            return None
    
    def _encode_clip(self, frames: list, output_path: Path) -> bool:
        """
        Encode frames to video file.
        
        Uses FFmpeg with hardware acceleration when available.
        
        Args:
            frames: List of FrameData objects
            output_path: Output file path
            
        Returns:
            True if successful
        """
        if not frames:
            return False
        
        # Get frame dimensions
        height, width = frames[0].frame.shape[:2]
        
        # Try hardware-accelerated encoding first (Jetson NVENC)
        if self._try_nvenc_encode(frames, output_path, width, height):
            return True
        
        # Fallback to OpenCV VideoWriter
        return self._opencv_encode(frames, output_path, width, height)
    
    def _try_nvenc_encode(self, frames: list, output_path: Path, width: int, height: int) -> bool:
        """
        Try hardware-accelerated encoding with FFmpeg + NVENC.
        
        Args:
            frames: List of FrameData objects
            output_path: Output file path
            width: Frame width
            height: Frame height
            
        Returns:
            True if successful
        """
        try:
            # Check if nvenc is available
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            has_nvenc = "hevc_nvenc" in result.stdout or "h264_nvenc" in result.stdout
            
            if not has_nvenc:
                return False
            
            # Determine encoder
            if self.codec == "h265" and "hevc_nvenc" in result.stdout:
                encoder = "hevc_nvenc"
            elif "h264_nvenc" in result.stdout:
                encoder = "h264_nvenc"
            else:
                return False
            
            # Write frames to temporary raw video
            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
                tmp_path = tmp.name
                for fd in frames:
                    # Convert BGR to YUV420 would be ideal, but for simplicity use raw
                    tmp.write(fd.frame.tobytes())
            
            try:
                # Encode with FFmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-pixel_format", "bgr24",
                    "-video_size", f"{width}x{height}",
                    "-framerate", str(self.fps),
                    "-i", tmp_path,
                    "-c:v", encoder,
                    "-preset", "fast",
                    "-rc", "vbr",
                    "-cq", str(self.crf),
                    "-an",  # No audio
                    str(output_path)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=60
                )
                
                if result.returncode == 0 and output_path.exists():
                    return True
                    
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            
            return False
            
        except Exception as e:
            logger.debug(f"NVENC encoding failed: {e}")
            return False
    
    def _opencv_encode(self, frames: list, output_path: Path, width: int, height: int) -> bool:
        """
        Encode frames using OpenCV VideoWriter.
        
        Args:
            frames: List of FrameData objects
            output_path: Output file path
            width: Frame width
            height: Frame height
            
        Returns:
            True if successful
        """
        try:
            # Use H.264 codec (more widely supported)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (width, height)
            )
            
            if not writer.isOpened():
                logger.error("Failed to open VideoWriter")
                return False
            
            try:
                for fd in frames:
                    writer.write(fd.frame)
                return True
            finally:
                writer.release()
                
        except Exception as e:
            logger.error(f"OpenCV encoding failed: {e}")
            return False
    
    def _cleanup_old_clips(self) -> None:
        """Remove old clips if storage limits exceeded."""
        try:
            clips = sorted(
                self.buffer_path.glob("*.mp4"),
                key=lambda p: p.stat().st_mtime
            )
            
            # Check count limit
            while len(clips) > self.max_clips:
                oldest = clips.pop(0)
                oldest.unlink()
                logger.debug(f"Removed old clip: {oldest}")
            
            # Check storage limit
            total_size = sum(c.stat().st_size for c in clips)
            max_bytes = self.max_storage_mb * 1024 * 1024
            
            while total_size > max_bytes and clips:
                oldest = clips.pop(0)
                total_size -= oldest.stat().st_size
                oldest.unlink()
                logger.debug(f"Removed clip for storage: {oldest}")
                
        except Exception as e:
            logger.warning(f"Clip cleanup failed: {e}")
    
    def get_buffer_status(self) -> dict:
        """Get current buffer status."""
        with self._lock:
            frame_count = len(self._buffer)
            oldest_time = self._buffer[0].timestamp if self._buffer else 0
            newest_time = self._buffer[-1].timestamp if self._buffer else 0
        
        return {
            "frames_in_buffer": frame_count,
            "max_frames": self.max_frames,
            "buffer_duration_seconds": newest_time - oldest_time if frame_count > 0 else 0,
            "pending_captures": len(self._pending_captures),
            "stats": self.stats.copy()
        }
    
    def get_clip_list(self) -> list:
        """Get list of saved clips."""
        clips = []
        for clip_path in self.buffer_path.glob("*.mp4"):
            stat = clip_path.stat()
            clips.append({
                "filename": clip_path.name,
                "path": str(clip_path),
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        return sorted(clips, key=lambda c: c["created_at"], reverse=True)
