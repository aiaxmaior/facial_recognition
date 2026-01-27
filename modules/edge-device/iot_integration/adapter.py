"""
Adapter to bridge existing FaceAuthSystem with IoT integration.

This adapter allows the existing facial_recognition.py system to work
with the new IoT event transmission and enrollment sync features.

Usage:
    from facial_recognition import FaceAuthSystem
    from iot_integration.adapter import IoTAdapter
    
    # Existing system
    face_system = FaceAuthSystem()
    
    # Wrap with IoT adapter
    iot_adapter = IoTAdapter(
        face_system=face_system,
        device_id="cam-001",
        broker_url="https://iot-broker.example.com"
    )
    iot_adapter.start()
    
    # Use existing match_from_frame, events auto-transmit to IoT broker
    name, distance, is_match = iot_adapter.match_from_frame(frame, track_id="face_0")
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np

from .event_validator import EventValidator
from .iot_client import IoTClient, IoTClientConfig
from .image_utils import compress_image_for_event
from .schemas.event_schemas import FacialIdEvent, EventMetadata

logger = logging.getLogger(__name__)


class IoTAdapter:
    """
    Adapter that wraps FaceAuthSystem to add IoT event transmission.
    
    This provides a drop-in enhancement for the existing system:
    - Validates events across multiple frames before transmitting
    - Sends events to IoT broker with compressed images
    - Maintains backward compatibility with existing code
    """
    
    def __init__(
        self,
        face_system: "FaceAuthSystem",
        device_id: str,
        broker_url: str,
        api_key: str = None,
        enable_iot: bool = True,
        confirmation_frames: int = 5,
        consistency_threshold: float = 0.8,
        cooldown_seconds: int = 30,
    ):
        """
        Initialize IoT adapter.
        
        Args:
            face_system: Existing FaceAuthSystem instance
            device_id: Unique device identifier
            broker_url: IoT broker URL
            api_key: Optional API key for broker
            enable_iot: If False, just wraps system without IoT
            confirmation_frames: Frames to confirm identity
            consistency_threshold: Required consistency ratio
            cooldown_seconds: Cooldown between same-user events
        """
        self.face_system = face_system
        self.device_id = device_id
        self.enable_iot = enable_iot
        
        # IoT client
        if enable_iot:
            client_config = IoTClientConfig(
                device_id=device_id,
                broker_url=broker_url,
                api_key=api_key,
            )
            self.iot_client = IoTClient(client_config)
        else:
            self.iot_client = None
        
        # Event validator
        self.event_validator = EventValidator(
            device_id=device_id,
            confirmation_frames=confirmation_frames,
            consistency_threshold=consistency_threshold,
            cooldown_seconds=cooldown_seconds,
            distance_threshold=face_system.threshold,
            on_event_validated=self._on_event_validated,
        )
        
        # Mapping from name -> user_id (populated from employee_id if available)
        self._name_to_user_id: Dict[str, str] = {}
        self._build_name_mapping()
        
        # Frame counter for tracking
        self._frame_id = 0
        self._current_frame: Optional[np.ndarray] = None
        self._current_bbox: Optional[List[int]] = None
        
        # Callbacks
        self.on_event_sent: Optional[Callable[[FacialIdEvent], None]] = None
        
        logger.info(f"IoTAdapter initialized: device={device_id}, iot={enable_iot}")
    
    def _build_name_mapping(self):
        """Build mapping from name to user_id using employee_id if available."""
        try:
            enrolled = self.face_system.list_enrolled()
            for entry in enrolled:
                name = entry.get('name')
                # Use employee_id as user_id if available, otherwise use name
                user_id = entry.get('employee_id') or name
                if name and user_id:
                    self._name_to_user_id[name] = user_id
            
            logger.info(f"Built name mapping for {len(self._name_to_user_id)} enrolled faces")
        except Exception as e:
            logger.warning(f"Failed to build name mapping: {e}")
    
    def _get_user_id(self, name: str) -> str:
        """Convert name to user_id."""
        return self._name_to_user_id.get(name, name)
    
    def start(self):
        """Start IoT services."""
        if self.iot_client:
            self.iot_client.start()
        logger.info("IoT adapter started")
    
    def stop(self):
        """Stop IoT services."""
        if self.iot_client:
            self.iot_client.stop()
        logger.info("IoT adapter stopped")
    
    def refresh_mapping(self):
        """Refresh the name -> user_id mapping after enrollment changes."""
        self._build_name_mapping()
    
    def _on_event_validated(self, event: FacialIdEvent):
        """Called when event validator confirms an identity."""
        # Attach image if we have one
        if self._current_frame is not None and self.iot_client:
            self.iot_client.send_event(
                event,
                image=self._current_frame,
                face_bbox=self._current_bbox,
            )
        
        # Callback
        if self.on_event_sent:
            self.on_event_sent(event)
        
        logger.info(f"Event sent to IoT broker: {event.user_id}")
    
    # =========================================================================
    # Wrapped FaceAuthSystem methods with IoT event handling
    # =========================================================================
    
    def match_from_frame(
        self,
        frame: np.ndarray,
        track_id: str = "default",
        face_bbox: List[int] = None,
    ) -> Tuple[str, float, bool]:
        """
        Match a frame against the enrollment database (with IoT event handling).
        
        This wraps FaceAuthSystem.match_from_frame and adds:
        - Event validation across frames
        - IoT event transmission when identity is confirmed
        
        Args:
            frame: BGR numpy array
            track_id: Face tracking ID (for multi-face tracking)
            face_bbox: Optional face bounding box [x, y, w, h]
            
        Returns:
            Tuple of (matched_name, distance_score, is_match)
        """
        self._frame_id += 1
        self._current_frame = frame
        self._current_bbox = face_bbox
        
        # Use existing face system for recognition
        name, distance, is_match = self.face_system.match_from_frame(frame)
        
        # Process through event validator if it's a match
        if is_match and name not in ("No Face", "Error", "Unknown"):
            user_id = self._get_user_id(name)
            
            # This may trigger _on_event_validated if identity is confirmed
            self.event_validator.process_recognition(
                track_id=track_id,
                user_id=user_id,
                distance=distance,
                frame_id=self._frame_id,
                face_bbox=face_bbox,
            )
        
        return name, distance, is_match
    
    def match(self, target_image_path: str) -> Tuple[str, float, bool]:
        """Match a target image (passthrough to face_system)."""
        return self.face_system.match(target_image_path)
    
    def match_averaged(self, frames: List[np.ndarray]) -> Tuple[str, float, bool]:
        """Match using averaged embedding (passthrough to face_system)."""
        return self.face_system.match_averaged(frames)
    
    def enroll(self, name: str, image_paths: List[str]) -> Optional[np.ndarray]:
        """Enroll a person (passthrough + refresh mapping)."""
        result = self.face_system.enroll(name, image_paths)
        if result is not None:
            self.refresh_mapping()
        return result
    
    def list_enrolled(self) -> List[Dict]:
        """List enrolled faces (passthrough)."""
        return self.face_system.list_enrolled()
    
    def delete_enrolled(self, name: str) -> bool:
        """Delete an enrolled face (passthrough + refresh mapping)."""
        result = self.face_system.delete_enrolled(name)
        if result:
            self.refresh_mapping()
        return result
    
    def load_database(self, force_reload: bool = False) -> Dict[str, np.ndarray]:
        """Load database (passthrough)."""
        return self.face_system.load_database(force_reload)
    
    # =========================================================================
    # IoT-specific methods
    # =========================================================================
    
    def get_iot_stats(self) -> Dict:
        """Get IoT client statistics."""
        stats = {
            "device_id": self.device_id,
            "iot_enabled": self.enable_iot,
            "validator": self.event_validator.stats,
            "active_tracks": len(self.event_validator.get_active_tracks()),
            "cooldowns": self.event_validator.get_cooldown_status(),
        }
        
        if self.iot_client:
            stats["client"] = self.iot_client.get_stats()
        
        return stats
    
    def clear_cooldown(self, name: str):
        """Clear cooldown for a user (by name)."""
        user_id = self._get_user_id(name)
        self.event_validator.clear_cooldown(user_id)
    
    def reset_tracking(self):
        """Reset all face tracking state."""
        self.event_validator.reset()


class LiveStreamWithIoT:
    """
    Enhanced LiveStreamRecognizer with IoT event transmission.
    
    Drop-in replacement for LiveStreamRecognizer that adds IoT features.
    
    Usage:
        from facial_recognition import FaceAuthSystem
        from iot_integration.adapter import LiveStreamWithIoT
        
        system = FaceAuthSystem()
        stream = LiveStreamWithIoT(
            face_system=system,
            device_id="cam-001",
            broker_url="https://iot-broker.example.com"
        )
        stream.start()
    """
    
    def __init__(
        self,
        face_system: "FaceAuthSystem",
        device_id: str,
        broker_url: str,
        api_key: str = None,
        camera_index: int = 0,
        enable_iot: bool = True,
        recognition_interval: float = 0.5,
    ):
        """
        Initialize live stream with IoT.
        
        Args:
            face_system: Existing FaceAuthSystem instance
            device_id: Unique device identifier
            broker_url: IoT broker URL
            api_key: Optional API key
            camera_index: Camera to use
            enable_iot: Enable IoT transmission
            recognition_interval: Seconds between recognitions
        """
        self.camera_index = camera_index
        self.recognition_interval = recognition_interval
        
        # Create IoT adapter
        self.adapter = IoTAdapter(
            face_system=face_system,
            device_id=device_id,
            broker_url=broker_url,
            api_key=api_key,
            enable_iot=enable_iot,
        )
        
        # Stream state
        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_result = {"name": "Initializing...", "distance": 0, "is_match": False}
        self.frame_lock = threading.Lock()
        self.last_recognition_time = 0
    
    def start(self) -> str:
        """Start the live stream."""
        import cv2
        
        if self.running:
            return "Already running"
        
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            return "❌ Failed to open camera"
        
        # Start IoT services
        self.adapter.start()
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        return "✅ Live stream started with IoT"
    
    def stop(self) -> str:
        """Stop the live stream."""
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.adapter.stop()
        return "Stream stopped"
    
    def _capture_loop(self):
        """Main capture and recognition loop."""
        import cv2
        
        track_id = "live_0"  # Single face tracking for live stream
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)  # Mirror
            display_frame = frame.copy()
            
            # Run recognition periodically
            current_time = time.time()
            if current_time - self.last_recognition_time >= self.recognition_interval:
                self.last_recognition_time = current_time
                
                try:
                    # Use adapter which handles IoT events
                    name, distance, is_match = self.adapter.match_from_frame(
                        frame, 
                        track_id=track_id
                    )
                    self.current_result = {
                        "name": name, 
                        "distance": distance, 
                        "is_match": is_match
                    }
                except Exception as e:
                    self.current_result = {"name": "Error", "distance": 0, "is_match": False}
                    logger.error(f"Recognition error: {e}")
            
            # Draw result on frame
            result = self.current_result
            if result["name"] not in ("No Face", "Error"):
                if result["is_match"]:
                    color = (0, 255, 0)
                    text = f"MATCH: {result['name']} ({result['distance']:.3f})"
                else:
                    color = (0, 165, 255)
                    text = f"Unknown ({result['name']}: {result['distance']:.3f})"
                
                cv2.rectangle(display_frame, (10, 10), (630, 70), color, 2)
                cv2.putText(display_frame, text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                cv2.putText(display_frame, "No face detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
            
            # Add IoT status indicator
            if self.adapter.enable_iot:
                cv2.putText(display_frame, "IoT: ON", (540, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            with self.frame_lock:
                self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            time.sleep(0.033)
    
    def get_frame(self):
        """Get current frame for Gradio display."""
        import numpy as np
        
        with self.frame_lock:
            if self.current_frame is not None:
                result = self.current_result
                status = f"{result['name']} | Distance: {result['distance']:.4f} | Match: {result['is_match']}"
                return self.current_frame, status
        
        # Placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        return placeholder, "Waiting for camera..."
    
    def get_stats(self) -> Dict:
        """Get IoT and recognition statistics."""
        return self.adapter.get_iot_stats()
