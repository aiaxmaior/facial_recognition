#!/usr/bin/env python3
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

# Local imports
from video_buffer import VideoRingBuffer

# Configure logging
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
        self.detector_backend = config.get("detector_backend", "yolov8")
        self.distance_threshold = config.get("distance_threshold", 0.35)
        self.min_face_size = config.get("min_face_size", 40)  # Lower default for sub-stream
        self.min_confidence = config.get("min_confidence", 0.4)  # Configurable confidence
        
        self._initialized = False
        self._embeddings_db: Dict[str, np.ndarray] = {}
        
        logger.info(f"FaceRecognitionPipeline configured: model={self.model_name}, detector={self.detector_backend}")
    
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
            
            # Log raw detections for debugging
            if faces:
                logger.info(f"[DETECTOR] Raw detections: {len(faces)} face(s)")
            
            # Filter by minimum face size
            valid_faces = []
            for i, face in enumerate(faces):
                conf = face.get("confidence", 0)
                region = face.get("facial_area", {})
                w = region.get("w", 0)
                h = region.get("h", 0)
                
                # Log each detection for debugging
                if conf > 0.2:  # Log anything above 20%
                    logger.info(f"[DETECTOR] Face {i}: conf={conf:.2f}, size={w}x{h}")
                
                if conf > self.min_confidence:
                    if w >= self.min_face_size and h >= self.min_face_size:
                        valid_faces.append(face)
                    else:
                        logger.info(f"[DETECTOR] Face {i} filtered: too small ({w}x{h} < {self.min_face_size})")
                else:
                    logger.info(f"[DETECTOR] Face {i} filtered: low confidence ({conf:.2f} < {self.min_confidence})")
            
            return valid_faces
            
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
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
            
            # Find closest match
            best_match = None
            best_distance = float('inf')
            
            for user_id, db_embedding in self._embeddings_db.items():
                # Cosine distance
                distance = 1 - np.dot(query_embedding, db_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = user_id
            
            if best_distance <= self.distance_threshold:
                return (best_match, best_distance)
            
            return None
            
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> list:
        """
        Process a single frame for face detection and recognition.
        
        Args:
            frame: BGR image frame
            frame_id: Frame sequence number
            
        Returns:
            List of (track_id, user_id, distance, bbox, face_img) tuples
            user_id is None for unrecognized faces
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        results = []
        faces = self.detect_faces(frame)
        
        if faces:
            logger.info(f"[DETECT] Frame {frame_id}: {len(faces)} face(s) detected")
        
        for i, face in enumerate(faces):
            track_id = f"face_{i}"  # Simple tracking by detection order
            region = face.get("facial_area", {})
            bbox = [
                region.get("x", 0),
                region.get("y", 0),
                region.get("w", 0),
                region.get("h", 0)
            ]
            
            # Get aligned face image
            face_img = face.get("face")
            if face_img is None:
                logger.debug(f"[DETECT] Face {i}: No aligned image available")
                continue
            
            # Convert to uint8 if needed
            if face_img.dtype != np.uint8:
                face_img = (face_img * 255).astype(np.uint8)
            
            # Recognize
            match = self.recognize_face(face_img)
            if match:
                user_id, distance = match
                logger.info(f"[MATCH] Face {i}: RECOGNIZED as {user_id} (distance={distance:.3f})")
                results.append((track_id, user_id, distance, bbox, face_img))
            else:
                # Unrecognized face - include with user_id=None
                logger.info(f"[MATCH] Face {i}: UNKNOWN (not in database)")
                results.append((track_id, None, 1.0, bbox, face_img))
        
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
    
    def __init__(self, config_path: str):
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
        
        # Send event to IoT broker (video clip sent separately when ready)
        self.iot_client.send_event(event)
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
                logger.info("Connected via GStreamer pipeline")
                return True
        except Exception as e:
            logger.debug(f"GStreamer failed: {e}")
        
        # Fallback to FFmpeg
        try:
            self.camera = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if self.camera.isOpened():
                logger.info("Connected via FFmpeg")
                return True
        except Exception as e:
            logger.debug(f"FFmpeg failed: {e}")
        
        # Last resort: default backend
        try:
            self.camera = cv2.VideoCapture(self.rtsp_url)
            if self.camera.isOpened():
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
    
    def _send_unknown_face_event(self, frame: np.ndarray, bbox: list, face_img: np.ndarray) -> None:
        """
        Send an error event for an unrecognized face.
        
        Args:
            frame: Full camera frame
            bbox: Face bounding box [x, y, w, h]
            face_img: Cropped/aligned face image
        """
        try:
            import cv2
            import base64
            
            # Encode face image as JPEG
            _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            face_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create error event
            from iot_integration.schemas.event_schemas import FaceRecognitionEvent
            
            event = FaceRecognitionEvent(
                device_id=self.device_id,
                person_name="UNKNOWN",
                person_id="UNKNOWN",
                confidence=0.0,
                metadata={
                    "error": "face_not_recognized",
                    "message": "Detected face not found in enrollment database",
                    "bbox": bbox,
                },
                debug=[{
                    "level": "warning",
                    "message": "Unrecognized face detected",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "frame_id": self._frame_id,
                }]
            )
            
            # Send with face image
            self.iot_client.send_event(event, image=face_img, face_bbox=bbox)
            logger.info(f"[EVENT] Sent unknown face event (image: {len(face_b64)} bytes)")
            
        except Exception as e:
            logger.error(f"[EVENT] Failed to send unknown face event: {e}")
    
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
        last_status_log = 0
        status_interval = 30  # Log status every 30 seconds
        
        logger.info("[LOOP] Main processing loop started")
        
        while self._running:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("[LOOP] Failed to read frame, reconnecting...")
                time.sleep(1)
                self._connect_camera()
                continue
            
            self._frame_id += 1
            self.stats["frames_processed"] += 1
            
            # Add frame to video buffer
            if self.video_buffer:
                self.video_buffer.add_frame(frame)
            
            # Process frame for faces
            results = self.recognition.process_frame(frame, self._frame_id)
            
            for track_id, user_id, distance, bbox, face_img in results:
                self.stats["faces_detected"] += 1
                
                if user_id is not None:
                    # Recognized face - normal flow
                    self.stats["recognitions"] += 1
                    logger.info(f"[EVENT] Recognized: {user_id} (distance={distance:.3f})")
                    
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
                else:
                    # Unrecognized face - send error event
                    logger.warning(f"[EVENT] Unrecognized face detected at frame {self._frame_id}")
                    self._send_unknown_face_event(frame, bbox, face_img)
            
            # Send heartbeat
            self._send_heartbeat()
            
            # Periodic status log
            if time.time() - last_status_log >= status_interval:
                fps = self.stats["frames_processed"] / max(time.time() - self.stats["start_time"], 1)
                logger.info(f"[STATUS] Frames: {self.stats['frames_processed']}, "
                           f"FPS: {fps:.1f}, "
                           f"Faces: {self.stats['faces_detected']}, "
                           f"Recognized: {self.stats['recognitions']}, "
                           f"Events: {self.stats['events_validated']}")
                last_status_log = time.time()
            
            # Frame rate control
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    
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
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check config exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        logger.info("Create config from template: /opt/qraie/config/device_config.json")
        sys.exit(1)
    
    # Create and start device
    device = EdgeDevice(args.config)
    
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
