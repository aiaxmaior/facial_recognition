"""
QRaie Edge Device - DeepStream Hybrid Pipeline

Facial Recognition Service using:
- DeepStream/GStreamer for hardware-accelerated video decode
- DeepFace for face detection and recognition (uses YOLOv8 backend)
- IoT broker for event transmission

Usage:
    python main_deepstream.py [--config CONFIG_PATH]
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
from typing import Optional, Dict, Any, List, NamedTuple

# Add system site-packages for gi (GObject) at END of path
# This lets venv packages take priority, but allows importing system libs not in venv
for syspath in ['/usr/lib/python3/dist-packages', '/usr/lib/python3.10/dist-packages']:
    if syspath not in sys.path:
        sys.path.append(syspath)

import cv2
import numpy as np

# Add parent directory to path for iot_integration imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iot_integration.iot_client import IoTClient, IoTClientConfig
from iot_integration.event_validator import EventValidator
from iot_integration.db_manager import EnrollmentDBManager
from iot_integration.schemas.event_schemas import FacialIdEvent

# Local imports
from ds_pipeline import DeepStreamPipeline, PipelineState
from face_recognition import ArcFaceRecognizer, RecognitionResult


# Simple detection result class (replaces TRT detector)
class FaceDetection(NamedTuple):
    """Face detection result."""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[np.ndarray] = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class HybridPipeline:
    """
    Hybrid DeepStream + Python face recognition pipeline.
    
    Architecture:
        DeepStream (RTSP → HW decode) 
            → TensorRT (YOLOv8 face detection)
            → ArcFace (face recognition)
            → Event Validator → IoT Client
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hybrid pipeline.
        
        Args:
            config: Full configuration dict
        """
        self.config = config
        self.device_id = config.get("device_id", "unknown-device")
        
        # Get config sections
        detection_config = config.get("detection", {})
        recognition_config = config.get("recognition", {})
        validation_config = config.get("validation", {})
        
        # Initialize components
        logger.info("Initializing hybrid pipeline components...")
        
        # 1. DeepStream video pipeline
        self.video_pipeline = DeepStreamPipeline(config)
        
        # 2. Detection config (using DeepFace's built-in detector)
        # Valid backends: opencv, ssd, mtcnn, retinaface, yunet, yolov8n, yolov8m, yolov8l, yolov11n, etc.
        # Note: ssd is faster and more stable with GStreamer
        self.detector_backend = detection_config.get("detector_backend", "ssd")
        self.conf_threshold = detection_config.get("conf_threshold", 0.5)
        self._deepface = None  # Lazy load
        
        # 3. ArcFace recognizer
        self.recognizer = ArcFaceRecognizer(
            model_name=recognition_config.get("model", "ArcFace"),
            distance_threshold=recognition_config.get("distance_threshold", 0.55),
            min_face_size=recognition_config.get("min_face_size", 40)
        )
        
        # 4. Event validator
        self.validator = EventValidator(
            device_id=self.device_id,
            confirmation_frames=validation_config.get("confirmation_frames", 3),
            consistency_threshold=validation_config.get("consistency_threshold", 0.6),
            cooldown_seconds=validation_config.get("cooldown_seconds", 10),
            distance_threshold=recognition_config.get("distance_threshold", 0.55),
            on_event_validated=self._on_event_validated
        )
        
        # 5. IoT client
        iot_config = IoTClientConfig(
            device_id=self.device_id,
            broker_url=config.get("broker_url"),
            api_key=config.get("api_key")
        )
        self.iot_client = IoTClient(iot_config)
        
        # 6. Enrollment database
        sync_config = config.get("sync", {})
        db_path = sync_config.get("enrollment_db_path")
        self.db_manager = EnrollmentDBManager(db_path) if db_path else None
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "faces_recognized": 0,
            "events_sent": 0,
            "start_time": None
        }
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._process_fps = config.get("recognition", {}).get("process_fps", 2)
        self._last_process_time = 0
        
        logger.info("Hybrid pipeline initialized")
    
    def _load_enrollments(self) -> None:
        """Load enrollment embeddings from database."""
        if not self.db_manager:
            logger.warning("No enrollment database configured")
            return
        
        try:
            embeddings = self.db_manager.get_all_embeddings()
            self.recognizer.load_embeddings(embeddings)
            logger.info(f"Loaded {len(embeddings)} enrollments")
        except Exception as e:
            logger.error(f"Failed to load enrollments: {e}")
    
    def _on_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Process frame from DeepStream pipeline.
        
        Called for every frame from the video pipeline.
        Rate-limited by _process_fps.
        """
        # Rate limiting
        now = time.time()
        if now - self._last_process_time < (1.0 / self._process_fps):
            return
        self._last_process_time = now
        
        try:
            self._process_frame(frame, timestamp)
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def _detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using DeepFace's built-in detector.
        
        Returns:
            List of FaceDetection objects
        """
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace
            logger.info(f"DeepFace loaded, using detector: {self.detector_backend}")
        
        try:
            # Use DeepFace.extract_faces for detection
            faces = self._deepface.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False
            )
            
            detections = []
            h, w = frame.shape[:2]
            
            # Debug: log first few detection results
            if not hasattr(self, '_debug_detect_count'):
                self._debug_detect_count = 0
            self._debug_detect_count += 1
            if self._debug_detect_count <= 5:
                logger.info(f"[DEBUG] Detection frame {self._debug_detect_count}: frame={frame.shape}, faces found={len(faces)}")
                if faces:
                    for i, f in enumerate(faces[:3]):
                        logger.info(f"  Face {i}: region={f.get('facial_area')}, conf={f.get('confidence')}")
                # Save first frame for debugging
                if self._debug_detect_count == 1:
                    cv2.imwrite("/tmp/debug_frame.jpg", frame)
                    logger.info(f"[DEBUG] Saved frame to /tmp/debug_frame.jpg")
            
            for face in faces:
                region = face.get("facial_area", {})
                conf = face.get("confidence", 1.0)
                
                # Skip bogus detections (full frame or zero confidence)
                face_w = region.get("w", 0)
                face_h = region.get("h", 0)
                if conf < 0.1 or face_w > h * 0.9 or face_h > w * 0.9:
                    continue
                
                if conf >= self.conf_threshold:
                    x1 = region.get("x", 0)
                    y1 = region.get("y", 0)
                    x2 = x1 + face_w
                    y2 = y1 + face_h
                    
                    detections.append(FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Process a single frame for detection and recognition."""
        with self._lock:
            self.stats["frames_processed"] += 1
        
        # 1. Detect faces with DeepFace
        detections = self._detect_faces(frame)
        
        if detections:
            with self._lock:
                self.stats["faces_detected"] += len(detections)
        
        # 2. Recognize each face
        results: List[RecognitionResult] = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Add margin for better recognition
            h, w = frame.shape[:2]
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(w, x2 + margin_x)
            crop_y2 = min(h, y2 + margin_y)
            
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if face_crop.size == 0:
                continue
            
            # Recognize
            result = self.recognizer.recognize(face_crop, det.bbox)
            if result:
                results.append(result)
                with self._lock:
                    self.stats["faces_recognized"] += 1
        
        # 3. Update validator with results
        for result in results:
            event = self.validator.process_recognition(
                track_id=f"face_{hash(result.bbox)}",
                user_id=result.user_id,
                distance=result.distance,
                frame_id=self.stats["frames_processed"],
                face_bbox=list(result.bbox),
                frame=frame
            )
            # Event is emitted via callback if validation passes
        
        # Periodic stats logging
        if self.stats["frames_processed"] % 100 == 0:
            self._log_stats()
    
    def _on_event_validated(self, event: FacialIdEvent) -> None:
        """Called when validator confirms an event."""
        logger.info(f"[EVENT] Validated: {event.identity_id} (confidence={event.confidence:.2f})")
        
        # Send to IoT broker
        try:
            self.iot_client.send_facial_id(event)
            with self._lock:
                self.stats["events_sent"] += 1
            logger.info(f"[EVENT] Sent to IoT broker")
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
    
    def _log_stats(self) -> None:
        """Log pipeline statistics."""
        with self._lock:
            elapsed = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
            fps = self.stats["frames_processed"] / elapsed if elapsed > 0 else 0
            
            video_stats = self.video_pipeline.get_stats()
            
            logger.info(
                f"[STATS] Processed: {self.stats['frames_processed']}, "
                f"Detected: {self.stats['faces_detected']}, "
                f"Recognized: {self.stats['faces_recognized']}, "
                f"Events: {self.stats['events_sent']}, "
                f"FPS: {fps:.1f}, "
                f"Video FPS: {video_stats.get('fps', 0):.1f}"
            )
    
    def start(self) -> None:
        """Start the pipeline."""
        logger.info("Starting hybrid pipeline...")
        
        # Initialize ArcFace
        if not self.recognizer.initialize():
            raise RuntimeError("Failed to initialize ArcFace recognizer")
        
        # Load enrollments
        self._load_enrollments()
        
        # Connect IoT client
        try:
            self.iot_client.connect()
            # Start heartbeat
            heartbeat_interval = self.config.get("heartbeat", {}).get("interval_seconds", 30)
            self.iot_client.start_heartbeat(heartbeat_interval)
        except Exception as e:
            logger.warning(f"IoT connection failed: {e}")
        
        # Reset stats
        self.stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "faces_recognized": 0,
            "events_sent": 0,
            "start_time": time.time()
        }
        
        # Start video pipeline
        self._running = True
        self.video_pipeline.start(self._on_frame)
        
        logger.info("Hybrid pipeline started")
    
    def stop(self) -> None:
        """Stop the pipeline."""
        logger.info("Stopping hybrid pipeline...")
        self._running = False
        
        # Stop video pipeline
        self.video_pipeline.stop()
        
        # Disconnect IoT
        try:
            self.iot_client.stop_heartbeat()
            self.iot_client.disconnect()
        except:
            pass
        
        # Final stats
        self._log_stats()
        logger.info("Hybrid pipeline stopped")
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running and self.video_pipeline.is_running()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QRaie Edge Device - DeepStream Pipeline")
    parser.add_argument(
        "--config", "-c",
        default="config/config.json",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(Path(__file__).parent.parent / config_path)
    
    logger.info(f"Loading config from: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Create pipeline
    pipeline = HybridPipeline(config)
    
    # Signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        pipeline.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    try:
        pipeline.start()
        
        logger.info("Pipeline running. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        # Give pipeline time to reach PLAYING state before checking
        time.sleep(2)
        
        while pipeline._running:
            time.sleep(1)
            
            # Check if video pipeline has failed
            if pipeline.video_pipeline.state.value == "error":
                logger.error("Video pipeline in error state")
                break
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
