"""
Example: Integrating IoT module with facial recognition.

This shows how to use the IoT integration module with the
existing facial recognition system.
"""

import logging
import time
import json
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import IoT integration components
from iot_integration import (
    DatabaseManager,
    EventValidator,
    IoTClient,
    IoTClientConfig,
    SyncManager,
    FacialIdEvent,
    compress_image_for_event,
)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


class IntegratedFaceRecognizer:
    """
    Example class showing integration of face recognition with IoT events.
    
    This wraps the existing FaceAuthSystem and adds:
    - Event validation across frames
    - IoT event transmission
    - Enrollment sync from central database
    """
    
    def __init__(self, config_path: str = None, config: dict = None):
        """
        Initialize integrated recognizer.
        
        Args:
            config_path: Path to config JSON file
            config: Config dict (alternative to file)
        """
        # Load config
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Must provide config_path or config")
        
        device_id = self.config["device_id"]
        broker_url = self.config["broker_url"]
        dev_mode = self.config.get("dev_mode", False)
        
        # Initialize database
        db_path = self.config.get("db_path", "enrolled_faces/faces_iot.db")
        self.db = DatabaseManager(db_path, dev_mode=dev_mode)
        self.db.initialize()
        
        # Initialize IoT client
        client_config = IoTClientConfig(
            device_id=device_id,
            broker_url=broker_url,
            api_key=self.config.get("api_key"),
            **self.config.get("event_queue", {}),
            **self.config.get("connection", {}),
            **{f"image_{k}": v for k, v in self.config.get("image_compression", {}).items()
               if k != "enabled"},
        )
        self.iot_client = IoTClient(client_config)
        
        # Initialize event validator
        validation_config = self.config.get("event_validation", {})
        self.event_validator = EventValidator(
            device_id=device_id,
            confirmation_frames=validation_config.get("confirmation_frames", 5),
            consistency_threshold=validation_config.get("consistency_threshold", 0.8),
            cooldown_seconds=validation_config.get("cooldown_seconds", 30),
            distance_threshold=validation_config.get("distance_threshold", 0.35),
            on_event_validated=self._on_event_validated,
        )
        
        # Initialize sync manager
        sync_config = self.config.get("sync", {})
        self.sync_manager = SyncManager(
            self.db,
            self.iot_client,
            model=sync_config.get("model", "ArcFace"),
            on_sync_complete=self._on_sync_complete,
        )
        
        # Local embedding cache (loaded from SQLite)
        self._embeddings_cache = {}
        self._cache_dirty = True
        
        # Recognition threshold
        self.distance_threshold = validation_config.get("distance_threshold", 0.35)
        
        logger.info(f"IntegratedFaceRecognizer initialized: device={device_id}")
    
    def start(self):
        """Start all services."""
        # Start IoT client
        self.iot_client.start()
        
        # Start periodic sync
        sync_interval = self.config.get("sync", {}).get("interval_minutes", 15)
        self.sync_manager.start_periodic_sync(interval_minutes=sync_interval)
        
        # Load initial embeddings
        self._refresh_embeddings_cache()
        
        logger.info("Services started")
    
    def stop(self):
        """Stop all services."""
        self.sync_manager.stop()
        self.iot_client.stop()
        logger.info("Services stopped")
    
    def _refresh_embeddings_cache(self):
        """Refresh embeddings from database."""
        self._embeddings_cache = self.db.get_all_enrollments()
        self._cache_dirty = False
        logger.info(f"Loaded {len(self._embeddings_cache)} embeddings")
    
    def _on_event_validated(self, event: FacialIdEvent):
        """Callback when event is validated."""
        logger.info(f"Event validated: {event.user_id}")
        # Event is automatically queued by validator callback
    
    def _on_sync_complete(self, additions: int, removals: int, version: int):
        """Callback when sync completes."""
        if additions > 0 or removals > 0:
            self._cache_dirty = True
            self._refresh_embeddings_cache()
    
    def process_frame(
        self,
        frame: np.ndarray,
        face_embedding: np.ndarray,
        track_id: str,
        frame_id: int,
        face_bbox: list = None,
    ) -> dict:
        """
        Process a frame with detected face.
        
        Args:
            frame: Video frame (BGR)
            face_embedding: Extracted face embedding
            track_id: Face tracking ID
            frame_id: Current frame number
            face_bbox: Face bounding box [x, y, w, h]
            
        Returns:
            Recognition result dict
        """
        # Refresh cache if needed
        if self._cache_dirty:
            self._refresh_embeddings_cache()
        
        # Match against database
        user_id, distance = self._match_embedding(face_embedding)
        
        is_match = distance <= self.distance_threshold
        confidence = 1.0 - distance if is_match else 0.0
        
        result = {
            "user_id": user_id,
            "distance": distance,
            "confidence": confidence,
            "is_match": is_match,
            "frame_id": frame_id,
        }
        
        # Process through event validator
        if is_match and user_id != "Unknown":
            event = self.event_validator.process_recognition(
                track_id=track_id,
                user_id=user_id,
                distance=distance,
                frame_id=frame_id,
                face_bbox=face_bbox,
            )
            
            if event:
                # Attach image and send to broker
                self.iot_client.send_event(
                    event,
                    image=frame,
                    face_bbox=face_bbox,
                )
                result["event_emitted"] = True
                result["event_id"] = event.event_id
        
        return result
    
    def _match_embedding(self, embedding: np.ndarray) -> tuple:
        """
        Match embedding against database.
        
        Args:
            embedding: Face embedding to match
            
        Returns:
            (user_id, distance) tuple
        """
        if not self._embeddings_cache:
            return "Unknown", 1.0
        
        # Cosine distance matching
        best_user = "Unknown"
        best_distance = 1.0
        
        # Normalize query embedding
        query_norm = embedding / np.linalg.norm(embedding)
        
        for user_id, db_embedding in self._embeddings_cache.items():
            # Normalize database embedding
            db_norm = db_embedding / np.linalg.norm(db_embedding)
            
            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(query_norm, db_norm)
            distance = 1.0 - similarity
            
            if distance < best_distance:
                best_distance = distance
                best_user = user_id
        
        return best_user, best_distance
    
    def get_status(self) -> dict:
        """Get system status."""
        return {
            "iot_client": self.iot_client.get_stats(),
            "sync": self.sync_manager.get_sync_status(),
            "validator": {
                "active_tracks": len(self.event_validator.get_active_tracks()),
                "stats": self.event_validator.stats,
            },
            "database": {
                "enrollment_count": len(self._embeddings_cache),
            },
        }


# =============================================================================
# Example Usage
# =============================================================================

def main():
    """Example main function."""
    
    # Example config (normally loaded from file)
    config = {
        "device_id": "cam-001",
        "broker_url": "https://iot-broker.example.com",
        "api_key": None,
        "db_path": "enrolled_faces/faces_iot.db",
        "dev_mode": True,  # Enable for testing
        
        "sync": {
            "interval_minutes": 15,
            "model": "ArcFace"
        },
        
        "event_validation": {
            "confirmation_frames": 5,
            "consistency_threshold": 0.8,
            "cooldown_seconds": 30,
            "distance_threshold": 0.35
        },
        
        "image_compression": {
            "enabled": True,
            "quality": 65,
            "max_size_kb": 50
        },
    }
    
    # Initialize
    recognizer = IntegratedFaceRecognizer(config=config)
    
    try:
        recognizer.start()
        
        # In real use, this would be your camera loop
        logger.info("System running. Press Ctrl+C to stop.")
        
        # Simulate some frames (in real use, these come from camera)
        # frame = camera.read()
        # embedding = extract_embedding(frame)
        # result = recognizer.process_frame(frame, embedding, "track_0", frame_id)
        
        # Keep running
        while True:
            time.sleep(1)
            status = recognizer.get_status()
            logger.debug(f"Status: {status}")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        recognizer.stop()


if __name__ == "__main__":
    main()
