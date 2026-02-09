"""
QRaie Edge Device - Main Entry Point

Facial Recognition Service for Jetson Orin Nano.
Captures video from RTSP camera, performs face detection and recognition,
validates events, and transmits to IoT broker.

Usage:
    python main.py [--config CONFIG_PATH]

Environment Variables:
    CUDA_VISIBLE_DEVICES: GPU device index (default: 0)
    CONFIG_PATH: Path to config file (default: /opt/qraie/config/device_config.json)
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

# Add parent directory to path for iot_integration imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iot_integration.iot_client import IoTClient, IoTClientConfig
from iot_integration.event_validator import EventValidator, EmotionEventValidator
from iot_integration.sync_manager import SyncManager
from iot_integration.db_manager import EnrollmentDBManager
from iot_integration.schemas.event_schemas import FacialIdEvent
from iot_integration.image_utils import encode_video_clip_b64
from iot_integration.logging_config import setup_logging, build_debug_entries

# Local imports
from video_buffer import VideoRingBuffer

# Configure basic logging (will be reconfigured in main())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    """
    Face detection and recognition pipeline.
    
    Uses DeepFace with ArcFace model for recognition.
    Optimized for Jetson with TensorRT acceleration when available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the face recognition pipeline.
        
        Args:
            config: Recognition configuration dict
        """
        self.config = config
        self.model_name = config.get("model", "ArcFace")
        self.detector_backend = config.get("detector_backend", "yunet")
        self.distance_threshold = config.get("distance_threshold", 0.35)
        self.min_face_size = config.get("min_face_size", 40)
        self.min_confidence = config.get("min_confidence", 0.5)
        
        self._initialized = False
        self._embeddings_db: Dict[str, np.ndarray] = {}
        self._detection_count = 0  # For periodic logging
        
        logger.info(f"FaceRecognitionPipeline configured: model={self.model_name}, detector={self.detector_backend}, min_conf={self.min_confidence}")
    
    def initialize(self) -> bool:
        """
        Initialize models (lazy loading).
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        try:
            # Import DeepFace (takes time to load models)
            logger.info("Loading face recognition models...")
            from deepface import DeepFace
            
            # Warm up the model with a dummy image
            dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
            try:
                DeepFace.represent(
                    dummy_img,
                    model_name=self.model_name,
                    detector_backend="skip",
                    enforce_detection=False
                )
            except Exception:
                pass  # Expected to fail with dummy image
            
            self._initialized = True
            logger.info("Face recognition models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize face recognition: {e}")
            return False
    
    def load_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        """
        Load enrollment embeddings into memory.
        
        Args:
            embeddings: Dict mapping user_id to embedding numpy array
        """
        self._embeddings_db = embeddings
        logger.info(f"Loaded {len(embeddings)} embeddings into recognition pipeline")
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of face detections with bounding boxes
        """
        try:
            from deepface import DeepFace
            
            faces = DeepFace.extract_faces(
                frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            self._detection_count += 1
            
            # Log raw detections periodically or when faces found with confidence
            if faces:
                has_confident = any(f.get("confidence", 0) > 0.1 for f in faces)
                if has_confident or self._detection_count % 100 == 0:
                    for i, f in enumerate(faces):
                        conf = f.get("confidence", 0)
                        area = f.get("facial_area", {})
                        logger.info(f"[DETECT] Face {i}: conf={conf:.3f}, area={area}")
            
            # Filter by confidence and minimum face size
            valid_faces = []
            for face in faces:
                conf = face.get("confidence", 0)
                if conf >= self.min_confidence:
                    region = face.get("facial_area", {})
                    w = region.get("w", 0)
                    h = region.get("h", 0)
                    if w >= self.min_face_size and h >= self.min_face_size:
                        valid_faces.append(face)
                        logger.info(f"[DETECT] Valid face: conf={conf:.3f}, size={w}x{h}")
            
            return valid_faces
            
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return []
    
    def recognize_face(self, face_img: np.ndarray) -> Optional[tuple]:
        """
        Recognize a face against enrolled embeddings.
        
        Args:
            face_img: Aligned face image
            
        Returns:
            (user_id, distance) if match found, None otherwise
        """
        if not self._embeddings_db:
            return None
        
        try:
            from deepface import DeepFace
            
            # Get embedding for detected face
            embedding_result = DeepFace.represent(
                face_img,
                model_name=self.model_name,
                detector_backend="skip",  # Already detected
                enforce_detection=False
            )
            
            if not embedding_result:
                return None
            
            query_embedding = np.array(embedding_result[0]["embedding"])
            
            # Normalize query embedding for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Find closest match using cosine distance
            # Note: DB embeddings should already be L2-normalized
            best_match = None
            best_distance = float('inf')
            
            for user_id, db_embedding in self._embeddings_db.items():
                # Normalize db_embedding if not already (for robustness)
                db_norm = np.linalg.norm(db_embedding)
                if db_norm > 0 and abs(db_norm - 1.0) > 0.01:
                    db_embedding = db_embedding / db_norm
                
                # Cosine distance = 1 - cosine_similarity
                # For normalized vectors: cosine_similarity = dot product
                similarity = np.dot(query_embedding, db_embedding)
                distance = 1.0 - similarity
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = user_id
            
            # Log best match for debugging
            if best_match:
                logger.debug(f"Best match: {best_match} with distance {best_distance:.4f} (threshold: {self.distance_threshold})")
            
            if best_distance <= self.distance_threshold:
                return (best_match, best_distance)
            
            return None
            
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray, frame_id: int, faces: list = None) -> list:
        """
        Process a single frame for face detection and recognition.
        
        Args:
            frame: BGR image frame (original resolution for quality recognition)
            frame_id: Frame sequence number
            faces: Pre-detected faces with scaled coordinates (optional, will detect if not provided)
            
        Returns:
            List of (track_id, user_id, distance, bbox) tuples
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        results = []
        if faces is None:
            faces = self.detect_faces(frame)
        
        h, w = frame.shape[:2]
        
        for i, face in enumerate(faces):
            track_id = f"face_{i}"  # Simple tracking by detection order
            region = face.get("facial_area", {})
            x = max(0, region.get("x", 0))
            y = max(0, region.get("y", 0))
            fw = region.get("w", 0)
            fh = region.get("h", 0)
            
            bbox = [x, y, fw, fh]
            
            # Crop face from original full-resolution frame for maximum recognition quality
            # This ensures faces at 15' distance have sufficient resolution for recognition
            # Add margin (20%) for better recognition context
            margin = int(max(fw, fh) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                logger.warning(f"[RECOGNIZE] Empty face crop at ({x},{y},{fw},{fh})")
                continue
            
            # Ensure minimum face size for recognition quality
            crop_h, crop_w = face_crop.shape[:2]
            min_crop_size = 80  # Minimum crop size for good recognition
            if crop_w < min_crop_size or crop_h < min_crop_size:
                logger.debug(f"[RECOGNIZE] Face crop too small: {crop_w}x{crop_h}, skipping recognition")
                continue
            
            # Recognize using the cropped face
            match = self.recognize_face(face_crop)
            if match:
                user_id, distance = match
                results.append((track_id, user_id, distance, bbox))
                logger.info(f"[MATCH] {user_id} (distance={distance:.3f})")
            else:
                # Log unrecognized face for debugging
                logger.debug(f"[NO MATCH] Face {i} not recognized (no match below threshold)")
        
        return results


class EdgeDevice:
    """
    Main edge device controller.
    
    Orchestrates:
    - Camera capture
    - Face detection/recognition
    - Event validation
    - Video buffering
    - IoT communication
    """
    
    def __init__(self, config_path: str, display: bool = False):
        """
        Initialize edge device.
        
        Args:
            config_path: Path to device configuration JSON
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Extract config sections
        self.device_id = self.config.get("device_id", "unknown-device")
        self.broker_url = self.config.get("broker_url")
        
        camera_config = self.config.get("camera", {})
        recognition_config = self.config.get("recognition", {})
        validation_config = self.config.get("validation", {})
        buffer_config = self.config.get("video_buffer", {})
        
        # Initialize components
        self.camera: Optional[cv2.VideoCapture] = None
        self.rtsp_url = camera_config.get("rtsp_url")
        self.target_fps = camera_config.get("fps", 25)
        
        # Face recognition pipeline
        self.recognition = FaceRecognitionPipeline(recognition_config)
        
        # Event validator
        self.validator = EventValidator(
            device_id=self.device_id,
            confirmation_frames=validation_config.get("confirmation_frames", 5),
            consistency_threshold=validation_config.get("consistency_threshold", 0.8),
            cooldown_seconds=validation_config.get("cooldown_seconds", 30),
            distance_threshold=recognition_config.get("distance_threshold", 0.35),
            on_event_validated=self._on_event_validated
        )
        
        # Video buffer
        self.video_buffer: Optional[VideoRingBuffer] = None
        if buffer_config.get("enabled", True):
            self.video_buffer = VideoRingBuffer(
                buffer_seconds=buffer_config.get("duration_seconds", 15),
                fps=self.target_fps,
                buffer_path=buffer_config.get("buffer_path", "/tmp/video_buffer"),
                pre_event_seconds=buffer_config.get("pre_event_seconds", 10),
                post_event_seconds=buffer_config.get("post_event_seconds", 5)
            )
            # Register callback for when clips are ready
            self.video_buffer.on_clip_ready = self._on_clip_ready
        
        # IoT client
        iot_config = IoTClientConfig(
            device_id=self.device_id,
            broker_url=self.broker_url,
            batch_size=10,
            batch_timeout_ms=5000
        )
        self.iot_client = IoTClient(iot_config)
        
        # Enrollment database
        sync_config = self.config.get("sync", {})
        db_path = sync_config.get("enrollment_db_path", "/opt/qraie/data/enrollments/enrollments.db")
        self.enrollment_db = EnrollmentDBManager(db_path)
        
        # State
        self._running = False
        self._frame_id = 0
        self._last_heartbeat = 0
        self._heartbeat_interval = self.config.get("heartbeat", {}).get("interval_seconds", 30)
        
        # Debug display
        self._display = display
        self._last_detections = []  # Store detections for drawing
        
        # Store frame/bbox/debug for event image capture
        self._pending_event_frame = None
        self._pending_event_bbox = None
        self._pending_event_debug = []
        
        # Processing settings
        self._process_fps = recognition_config.get("process_fps", 1)  # Process 1 frame per second
        # Detection resolution: high resolution (2560px default) for better distant face detection
        # Recognition still uses full-resolution crops from original frame
        # At 15ft with 90° FOV: face is ~42px in 2560px frame (meets 30px min_face_size threshold)
        # If camera is already 2560px, detection runs at full resolution (no downscaling)
        self._detection_width = recognition_config.get("detection_width", 2560)  # Detection resolution
        self._process_width = recognition_config.get("process_width", 640)  # Legacy: kept for backward compat
        self._last_process_time = 0
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "recognitions": 0,
            "events_validated": 0,
            "start_time": None
        }
        
        logger.info(f"EdgeDevice initialized: device_id={self.device_id}")
    
    def _load_config(self, path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return {}
    
    def _on_event_validated(self, event: FacialIdEvent) -> None:
        """Callback when an event is validated."""
        logger.info(f"Event validated: user={event.user_id}, confidence={event.confidence:.2f}")
        
        # Attach debug entries to event for Graylog
        if self._pending_event_debug:
            event.debug = self._pending_event_debug
        
        # Send event to IoT broker with face image (synchronous for proper logging)
        # Use stored frame and bbox from the recognition that triggered this event
        frame = self._pending_event_frame
        bbox = self._pending_event_bbox
        
        if frame is not None:
            success = self.iot_client.send_event_sync(
                event,
                image=frame,
                face_bbox=bbox
            )
            if success:
                logger.info(f"Event sent with face image")
            else:
                logger.error(f"Event transmission failed")
        else:
            # Fallback: send without image
            success = self.iot_client.send_event_sync(event)
            if success:
                logger.warning(f"Event sent without image (no frame available)")
            else:
                logger.error(f"Event transmission failed (no image)")
        
        self.stats["events_validated"] += 1
        
        # Trigger video clip capture if buffer enabled
        # Video will be sent via _on_clip_ready callback when encoding completes
        if self.video_buffer:
            # Clip filename is device_id + event_id
            clip_event_id = f"{self.device_id}_{event.event_id}"
            self.video_buffer.capture_event_clip(clip_event_id)
    
    def _on_clip_ready(self, event_id: str, clip_path: str) -> None:
        """
        Callback when a video clip is ready to send.
        
        Sends the video clip to IoT broker with:
        - story: VLM narrative description (TODO: integrate VLM)
        - debug: Graylog debug entries
        - metadata: Video metadata
        """
        logger.info(f"Video clip ready: {event_id}")
        
        try:
            # Encode video to base64 for transmission
            video_b64 = encode_video_clip_b64(clip_path)
            
            if not video_b64:
                logger.error(f"Failed to encode video clip: {clip_path}")
                return
            
            # TODO: Get VLM story when emotion event is triggered
            # For now, story is None - will be populated when VLM integration is complete
            # story = await vlm_service.analyze_video(clip_path)
            story = None  # Placeholder for VLM narrative
            
            # Build debug/Graylog entries
            debug_entries = [
                {
                    "level": "info",
                    "message": f"Video clip captured for event {event_id}",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "clip_path": clip_path,
                    "clip_size_kb": len(video_b64) / 1024
                }
            ]
            
            # Video metadata
            metadata = {
                "duration": self.video_buffer.pre_event_seconds + self.video_buffer.post_event_seconds,
                "fps": self.video_buffer.fps,
                "resolution": f"{self.video_buffer.resolution[0]}x{self.video_buffer.resolution[1]}",
            }
            
            # Send video clip to IoT broker with story and debug
            self.iot_client.send_video_clip(
                event_id=event_id,
                video_b64=video_b64,
                story=story,
                debug=debug_entries,
                metadata=metadata
            )
            logger.info(f"Video clip sent: {event_id} ({len(video_b64)/1024:.1f} KB)")
                
        except Exception as e:
            logger.error(f"Failed to send video clip {event_id}: {e}")
    
    def _connect_camera(self) -> bool:
        """Connect to RTSP camera."""
        if not self.rtsp_url:
            logger.error("No RTSP URL configured")
            return False
        
        logger.info(f"Connecting to camera: {self.rtsp_url.split('@')[-1]}")  # Hide password
        
        # Use GStreamer pipeline for better RTSP handling on Jetson
        # Capture at full resolution for better recognition quality
        # Detection will be done at detection_width (1280px) in software
        # Recognition uses full-resolution crops for maximum accuracy
        gst_pipeline = (
            f"rtspsrc location={self.rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink"
        )
        
        try:
            # Try GStreamer first (optimized for Jetson)
            self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if self.camera.isOpened():
                actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Connected via GStreamer pipeline @ {actual_w}x{actual_h} (full resolution)")
                return True
        except Exception as e:
            logger.debug(f"GStreamer failed: {e}")
        
        # Fallback to FFmpeg
        try:
            self.camera = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if self.camera.isOpened():
                # Capture at full resolution for better recognition quality
                # Don't downscale here - let detection pipeline handle it
                actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Connected via FFmpeg @ {actual_w}x{actual_h} (full resolution)")
                return True
        except Exception as e:
            logger.debug(f"FFmpeg failed: {e}")
        
        # Last resort: default backend
        try:
            self.camera = cv2.VideoCapture(self.rtsp_url)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._process_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._process_width * 9 / 16))
                logger.info("Connected via default backend")
                return True
        except Exception as e:
            logger.error(f"All camera backends failed: {e}")
        
        return False
    
    def _load_enrollments(self) -> None:
        """Load enrollment embeddings from database."""
        try:
            embeddings = self.enrollment_db.get_all_embeddings()
            self.recognition.load_embeddings(embeddings)
            logger.info(f"Loaded {len(embeddings)} enrollments")
        except Exception as e:
            logger.error(f"Failed to load enrollments: {e}")
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat to IoT broker."""
        current_time = time.time()
        if current_time - self._last_heartbeat < self._heartbeat_interval:
            return
        
        try:
            metrics = {
                "cpu_percent": self._get_cpu_percent(),
                "memory_percent": self._get_memory_percent(),
                "temperature_c": self._get_temperature(),
                "uptime_seconds": int(current_time - (self.stats["start_time"] or current_time))
            }
            
            cv_stats = {
                "fps_current": self._calculate_fps(),
                "frames_processed": self.stats["frames_processed"],
                "detections_count": self.stats["faces_detected"],
                "recognitions_count": self.stats["recognitions"]
            }
            
            self.iot_client.send_heartbeat(metrics=metrics, cv_stats=cv_stats)
            self._last_heartbeat = current_time
            
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
    
    def _get_cpu_percent(self) -> float:
        """Get CPU usage percentage."""
        try:
            with open('/proc/loadavg', 'r') as f:
                load = float(f.read().split()[0])
            return min(load * 100 / os.cpu_count(), 100.0)
        except Exception:
            return 0.0
    
    def _get_memory_percent(self) -> float:
        """Get memory usage percentage."""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            mem_info = {}
            for line in lines:
                parts = line.split(':')
                if len(parts) == 2:
                    mem_info[parts[0]] = int(parts[1].split()[0])
            total = mem_info.get('MemTotal', 1)
            available = mem_info.get('MemAvailable', 0)
            return (1 - available / total) * 100
        except Exception:
            return 0.0
    
    def _get_temperature(self) -> float:
        """Get GPU/CPU temperature."""
        try:
            # Jetson thermal zone
            with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0
            return temp
        except Exception:
            return 0.0
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        if not self.stats["start_time"]:
            return 0.0
        elapsed = time.time() - self.stats["start_time"]
        if elapsed > 0:
            return self.stats["frames_processed"] / elapsed
        return 0.0
    
    def _draw_debug_overlay(self, frame: np.ndarray, detections: list, raw_faces: list = None) -> np.ndarray:
        """
        Draw debug overlay with bounding boxes and info.
        
        Args:
            frame: BGR image frame
            detections: List of (track_id, user_id, distance, bbox) tuples from recognition
            raw_faces: Raw face detections from detector (before recognition)
            
        Returns:
            Frame with overlay drawn
        """
        # Resize frame for display if too large
        h, w = frame.shape[:2]
        display_scale = 1.0
        if w > 1920:
            display_scale = 1920 / w
            display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
            h, w = display_frame.shape[:2]
        else:
            display_frame = frame.copy()
        
        # Draw raw detections in yellow (faces detected but not necessarily recognized)
        if raw_faces:
            for i, face in enumerate(raw_faces):
                region = face.get("facial_area", {})
                x = int(region.get("x", 0) * display_scale)
                y = int(region.get("y", 0) * display_scale)
                fw = int(region.get("w", 0) * display_scale)
                fh = int(region.get("h", 0) * display_scale)
                conf = face.get("confidence", 0)
                
                # Yellow box for raw detection
                cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), (0, 255, 255), 2)
                label = f"conf={conf:.2f}"
                cv2.putText(display_frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw recognized faces in green with user info
        for track_id, user_id, distance, bbox in detections:
            x = int(bbox[0] * display_scale)
            y = int(bbox[1] * display_scale)
            bw = int(bbox[2] * display_scale)
            bh = int(bbox[3] * display_scale)
            
            # Green box for recognized face
            cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            
            # Label with user ID and confidence
            confidence_pct = (1 - distance) * 100
            label = f"{user_id}"
            sublabel = f"{confidence_pct:.0f}% match"
            
            # Draw name label with background
            font_scale = 0.8
            thickness = 2
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            sublabel_size, _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background rectangle
            padding = 5
            bg_width = max(label_size[0], sublabel_size[0]) + padding * 2
            bg_height = label_size[1] + sublabel_size[1] + padding * 3
            
            cv2.rectangle(display_frame, (x, y - bg_height), (x + bg_width, y), (0, 200, 0), -1)
            cv2.rectangle(display_frame, (x, y - bg_height), (x + bg_width, y), (0, 255, 0), 2)
            
            # Draw text
            cv2.putText(display_frame, label, (x + padding, y - sublabel_size[1] - padding * 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            cv2.putText(display_frame, sublabel, (x + padding, y - padding),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        
        # Draw stats overlay
        fps = self._calculate_fps()
        enrolled_count = len(self.recognition._embeddings_db) if hasattr(self.recognition, '_embeddings_db') else 0
        
        stats_text = [
            f"Device: {self.device_id}",
            f"FPS: {fps:.1f}",
            f"Frames: {self.stats['frames_processed']}",
            f"Enrolled: {enrolled_count}",
            f"Detected: {self.stats['faces_detected']}",
            f"Events: {self.stats['events_validated']}",
        ]
        
        # Semi-transparent background for stats
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (220, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        for i, text in enumerate(stats_text):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(display_frame, text, (15, 30 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        
        # Status indicator
        if enrolled_count == 0:
            status = "NO ENROLLMENTS"
            status_color = (0, 0, 255)  # Red
        elif raw_faces:
            status = f"TRACKING {len(raw_faces)} FACE(S)"
            status_color = (0, 255, 255)  # Yellow
        else:
            status = "SCANNING..."
            status_color = (200, 200, 200)  # Gray
        
        cv2.putText(display_frame, status, (w - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Instructions
        cv2.putText(display_frame, "Press 'q' to quit | 's' to save frame", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return display_frame
    
    def start(self) -> None:
        """Start the edge device processing loop."""
        logger.info("Starting EdgeDevice...")
        
        # Initialize recognition pipeline
        if not self.recognition.initialize():
            logger.error("Failed to initialize recognition pipeline")
            return
        
        # Load enrollments
        self._load_enrollments()
        
        # Connect camera
        if not self._connect_camera():
            logger.error("Failed to connect to camera")
            return
        
        # Start IoT client
        self.iot_client.start()
        
        # Start video buffer
        if self.video_buffer:
            self.video_buffer.start()
        
        self._running = True
        self.stats["start_time"] = time.time()
        
        logger.info("EdgeDevice started - entering main loop")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def _main_loop(self) -> None:
        """Main processing loop."""
        frame_time = 1.0 / self.target_fps
        logger.info("[LOOP] Main processing loop started")
        
        # Create window if display enabled
        if self._display:
            cv2.namedWindow("QRaie Face Recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("QRaie Face Recognition", 1280, 720)
            logger.info("[DISPLAY] Debug display window opened")
        
        while self._running:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("Failed to read frame, reconnecting...")
                time.sleep(1)
                self._connect_camera()
                continue
            
            self._frame_id += 1
            self.stats["frames_processed"] += 1
            
            # Add frame to video buffer
            if self.video_buffer:
                self.video_buffer.add_frame(frame)
            
            # Rate-limit face processing to reduce CPU load
            current_time = time.time()
            process_interval = 1.0 / self._process_fps
            should_process = (current_time - self._last_process_time) >= process_interval
            
            raw_faces = []
            results = []
            
            if should_process:
                self._last_process_time = current_time
                
                # Optimized two-stage approach:
                # 1. Detect at high resolution (2560px default) for better distant face detection
                #    At 15ft with 90° FOV: face is ~42px in 2560px frame
                # 2. Recognize from full-resolution crops for maximum accuracy
                orig_h, orig_w = frame.shape[:2]
                
                # Use detection_width (default 2560px) for detection to catch distant faces
                # If camera is already at this resolution, no downscaling occurs (best for 15ft)
                # If camera is higher res, we downscale but preserve face size for detection
                detection_scale = self._detection_width / max(orig_w, orig_h)
                
                if detection_scale < 1.0:
                    # Downscale for detection (but keep higher than 640px for distant faces)
                    detect_frame = cv2.resize(frame, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_LINEAR)
                else:
                    # Frame is smaller than detection_width, use as-is
                    detect_frame = frame
                    detection_scale = 1.0
                
                # Benchmark: Detect faces on detection-resolution frame
                t0 = time.time()
                raw_faces_detect = self.recognition.detect_faces(detect_frame)
                t_detect = time.time() - t0
                
                # Scale face coordinates back to original frame size
                # When detection_scale < 1.0, coordinates need to be scaled up by (1/detection_scale)
                raw_faces = []
                if detection_scale > 0:
                    inv_scale = 1.0 / detection_scale
                    for face in raw_faces_detect:
                        conf = face.get("confidence", 0)
                        region = face.get("facial_area", {})
                        # Scale coordinates back up to original frame coordinates
                        scaled_region = {
                            "x": int(region.get("x", 0) * inv_scale),
                            "y": int(region.get("y", 0) * inv_scale),
                            "w": int(region.get("w", 0) * inv_scale),
                            "h": int(region.get("h", 0) * inv_scale),
                        }
                        # Copy other fields
                        scaled_face = face.copy()
                        scaled_face["facial_area"] = scaled_region
                        raw_faces.append(scaled_face)
                else:
                    raw_faces = raw_faces_detect
                
                if raw_faces:
                    logger.info(f"[DETECT] Found {len(raw_faces)} face(s) in {t_detect*1000:.0f}ms @ {detect_frame.shape[1]}x{detect_frame.shape[0]} (orig: {orig_w}x{orig_h})")
                    for i, face in enumerate(raw_faces):
                        conf = face.get("confidence", 0)
                        region = face.get("facial_area", {})
                        logger.info(f"[DETECT] Face {i}: conf={conf:.2f}, bbox=({region['x']},{region['y']},{region['w']},{region['h']})")
                
                # Benchmark: Recognition on original frame (for quality)
                t1 = time.time()
                results = self.recognition.process_frame(frame, self._frame_id, faces=raw_faces)
                t_recognize = time.time() - t1
                
                # Log recognition results
                if results:
                    for track_id, user_id, distance, bbox in results:
                        logger.info(f"[RECOGNIZE] {user_id}: distance={distance:.3f}, match={'YES' if distance <= self.recognition.distance_threshold else 'NO'}")
                
                logger.debug(f"[PERF] Detect: {t_detect*1000:.0f}ms ({detect_frame.shape[1]}x{detect_frame.shape[0]}), Recognize: {t_recognize*1000:.0f}ms, Total: {(t_detect+t_recognize)*1000:.0f}ms")
                
                # Cache for display
                self._last_detections = (raw_faces, results)
            
            for track_id, user_id, distance, bbox in results:
                self.stats["faces_detected"] += 1
                self.stats["recognitions"] += 1
                
                # Store frame, bbox, and debug entries for potential event (used in callback)
                self._pending_event_frame = frame.copy()
                self._pending_event_bbox = bbox
                self._pending_event_debug = build_debug_entries(
                    frame_id=self._frame_id,
                    detection_time_ms=t_detect * 1000 if should_process else None,
                    recognition_time_ms=t_recognize * 1000 if should_process else None,
                    faces_detected=len(raw_faces) if raw_faces else 0,
                    pipeline_state="recognition",
                    extra={
                        "distance": round(distance, 4),
                        "track_id": track_id,
                    }
                )
                
                # Validate event
                event = self.validator.process_recognition(
                    track_id=track_id,
                    user_id=user_id,
                    distance=distance,
                    frame_id=self._frame_id,
                    face_bbox=bbox,
                    frame=frame
                )
                # Event handling done in callback
            
            # Display debug frame
            if self._display:
                # Use cached detections if we didn't process this frame
                if not should_process and hasattr(self, '_last_detections') and self._last_detections:
                    raw_faces, results = self._last_detections
                
                display_frame = self._draw_debug_overlay(frame, results, raw_faces)
                cv2.imshow("QRaie Face Recognition", display_frame)
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("[DISPLAY] Quit requested via keyboard")
                    self._running = False
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"data/frame_{self._frame_id}.jpg"
                    cv2.imwrite(save_path, frame)
                    logger.info(f"[DISPLAY] Frame saved to {save_path}")
            
            # Log status periodically
            if self._frame_id % 300 == 0:  # Every ~30 seconds at 10fps
                fps = self._calculate_fps()
                logger.info(f"[STATUS] Frames: {self.stats['frames_processed']}, FPS: {fps:.1f}, "
                           f"Faces: {self.stats['faces_detected']}, Recognized: {self.stats['recognitions']}, "
                           f"Events: {self.stats['events_validated']}")
            
            # Send heartbeat
            self._send_heartbeat()
            
            # Frame rate control
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        # Cleanup display
        if self._display:
            cv2.destroyAllWindows()
    
    def stop(self) -> None:
        """Stop the edge device."""
        logger.info("Stopping EdgeDevice...")
        self._running = False
        
        # Stop components
        if self.video_buffer:
            self.video_buffer.stop()
        
        self.iot_client.stop()
        
        if self.camera:
            self.camera.release()
        
        # Print statistics
        elapsed = time.time() - (self.stats["start_time"] or time.time())
        logger.info(f"EdgeDevice stopped. Stats: "
                   f"frames={self.stats['frames_processed']}, "
                   f"fps={self.stats['frames_processed']/max(elapsed,1):.1f}, "
                   f"events={self.stats['events_validated']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QRaie Edge Device - Facial Recognition")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "/opt/qraie/config/device_config.json"),
        help="Path to device configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live video with bounding boxes (requires display or X11 forwarding)"
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        default=True,
        help="Enable JSON structured logging (default: enabled)"
    )
    parser.add_argument(
        "--no-json-logs",
        action="store_true",
        help="Disable JSON structured logging"
    )
    
    args = parser.parse_args()
    
    # Check config exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("Create config from template: /opt/qraie/config/device_config.json")
        sys.exit(1)
    
    # Load config to get device_id for logging
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        device_id = config.get("device_id", "unknown")
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    # Log dir: always under project root (edge-device) so we get a log file regardless of cwd
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    
    # Set up structured logging (always write to file)
    log_level = logging.DEBUG if args.debug else logging.INFO
    use_json = args.json_logs and not args.no_json_logs
    
    setup_logging(
        device_id=device_id,
        log_dir=str(log_dir),
        console_level=log_level,
        file_level=logging.DEBUG,
        json_logs=use_json,
    )
    
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / (f"events_{date_str}.jsonl" if use_json else f"device_{date_str}.log")
    logger.info(
        f"Logging initialized: device_id={device_id}, json={use_json}, level={log_level}. "
        f"Log file: {log_file}"
    )
    
    # Create and start device
    device = EdgeDevice(args.config, display=args.display)
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        device.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start processing
    device.start()


if __name__ == "__main__":
    main()
