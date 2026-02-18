"""
Event Validator with Temporal Tracking.

Implements the event validation logic that tracks individuals across
multiple consecutive frames before emitting a validated event.

This ensures that:
1. Random false positives don't trigger events
2. Identity is consistent across N frames
3. Same user doesn't trigger duplicate events within cooldown period
"""

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Callable
from threading import Lock

from .schemas.event_schemas import FacialIdEvent, EmotionEvent, EventMetadata

logger = logging.getLogger(__name__)


@dataclass
class TrackingBuffer:
    """Buffer for tracking recognition results for a single face track."""
    track_id: str
    recognitions: List[Dict] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    validated: bool = False
    emitted: bool = False


@dataclass
class RecognitionResult:
    """Single frame recognition result."""
    user_id: str
    distance: float
    confidence: float
    frame_id: int
    timestamp: float
    face_bbox: Optional[List[int]] = None


class EventValidator:
    """
    Validates recognition events by tracking individuals across frames.
    
    Events are only emitted when:
    1. Same user_id is detected in N consecutive frames
    2. Consistency ratio meets threshold (e.g., 80% same ID)
    3. User hasn't triggered an event within cooldown period
    
    State Machine:
        Detected -> Tracking -> Validated -> EventEmit -> Cooldown
        
    Usage:
        validator = EventValidator(
            device_id="cam-001",
            confirmation_frames=5,
            consistency_threshold=0.8,
            cooldown_seconds=30
        )
        
        # For each frame with a recognition:
        event = validator.process_recognition(
            track_id="face_0",
            user_id="EMP-123",
            distance=0.28,
            frame_id=100
        )
        
        if event:
            # Event is validated, send to IoT broker
            iot_client.send_event(event)
    """
    
    def __init__(
        self,
        device_id: str,
        confirmation_frames: int = 5,
        consistency_threshold: float = 0.8,
        cooldown_seconds: int = 30,
        distance_threshold: float = 0.35,
        max_track_age_seconds: float = 5.0,
        on_event_validated: Callable[[FacialIdEvent], None] = None,
    ):
        """
        Initialize event validator.
        
        Args:
            device_id: Device identifier for events
            confirmation_frames: Frames required to confirm identity
            consistency_threshold: Required ratio of consistent IDs (0.0-1.0)
            cooldown_seconds: Seconds before same user can trigger new event
            distance_threshold: Maximum distance for valid recognition
            max_track_age_seconds: Max age before track is reset
            on_event_validated: Callback when event is validated
        """
        self.device_id = device_id
        self.confirmation_frames = confirmation_frames
        self.consistency_threshold = consistency_threshold
        self.cooldown_seconds = cooldown_seconds
        self.distance_threshold = distance_threshold
        self.max_track_age = max_track_age_seconds
        self.on_event_validated = on_event_validated
        
        # Tracking state
        self._tracks: Dict[str, TrackingBuffer] = {}
        self._user_cooldowns: Dict[str, float] = {}  # user_id -> last_event_time
        self._lock = Lock()
        
        # Statistics
        self.stats = {
            "recognitions_processed": 0,
            "events_validated": 0,
            "events_rejected_consistency": 0,
            "events_rejected_cooldown": 0,
            "tracks_created": 0,
            "tracks_expired": 0,
        }
    
    def process_recognition(
        self,
        track_id: str,
        user_id: str,
        distance: float,
        frame_id: int,
        face_bbox: List[int] = None,
        frame: "np.ndarray" = None,
    ) -> Optional[FacialIdEvent]:
        """
        Process a single recognition result.
        
        Args:
            track_id: Face tracking ID (from face tracker)
            user_id: Recognized user ID
            distance: Cosine distance score
            frame_id: Current frame number
            face_bbox: Face bounding box [x, y, w, h]
            frame: Original frame (for event image capture)
            
        Returns:
            FacialIdEvent if validation passes, None otherwise
        """
        current_time = time.time()
        self.stats["recognitions_processed"] += 1
        
        # Skip if distance exceeds threshold (not a valid recognition)
        if distance > self.distance_threshold:
            return None
        
        confidence = 1.0 - distance
        
        with self._lock:
            # Clean up old tracks
            self._cleanup_stale_tracks(current_time)
            
            # Get or create tracking buffer
            if track_id not in self._tracks:
                self._tracks[track_id] = TrackingBuffer(track_id=track_id)
                self.stats["tracks_created"] += 1
            
            track = self._tracks[track_id]
            track.last_seen = current_time
            
            # Add recognition to buffer
            track.recognitions.append({
                "user_id": user_id,
                "distance": distance,
                "confidence": confidence,
                "frame_id": frame_id,
                "timestamp": current_time,
                "face_bbox": face_bbox,
            })
            
            # Check if we have enough frames
            if len(track.recognitions) < self.confirmation_frames:
                logger.info(f"[VALIDATOR] {track_id}: {user_id} frame {len(track.recognitions)}/{self.confirmation_frames} (dist={distance:.3f})")
                return None
            
            # Already emitted for this track?
            if track.emitted:
                return None
            
            # Check consistency
            validated_user, avg_distance, avg_confidence = self._check_consistency(track)
            
            if validated_user is None:
                self.stats["events_rejected_consistency"] += 1
                return None
            
            # Check cooldown
            if not self._check_cooldown(validated_user, current_time):
                self.stats["events_rejected_cooldown"] += 1
                return None
            
            # Event validated!
            track.validated = True
            track.emitted = True
            self._user_cooldowns[validated_user] = current_time
            self.stats["events_validated"] += 1
            
            # Create event
            event = FacialIdEvent(
                device_id=self.device_id,
                user_id=validated_user,
                confidence=avg_confidence,
                metadata=EventMetadata(
                    distance=avg_distance,
                    frames_tracked=len(track.recognitions),
                    face_bbox=face_bbox,
                ),
            )
            
            logger.info(
                f"Event validated: user={validated_user}, "
                f"confidence={avg_confidence:.2f}, frames={len(track.recognitions)}"
            )
            
            # Callback if registered
            if self.on_event_validated:
                try:
                    self.on_event_validated(event)
                except Exception as e:
                    logger.error(f"Event callback failed: {e}")
            
            return event
    
    def _check_consistency(self, track: TrackingBuffer) -> tuple:
        """
        Check if recognitions in track are consistent.
        
        Returns:
            (user_id, avg_distance, avg_confidence) if consistent, (None, 0, 0) otherwise
        """
        if not track.recognitions:
            return None, 0, 0
        
        # Count occurrences of each user_id
        user_counts = defaultdict(int)
        user_distances = defaultdict(list)
        
        for rec in track.recognitions[-self.confirmation_frames:]:
            uid = rec["user_id"]
            user_counts[uid] += 1
            user_distances[uid].append(rec["distance"])
        
        # Find most common user
        total = sum(user_counts.values())
        best_user = None
        best_count = 0
        
        for user_id, count in user_counts.items():
            if count > best_count:
                best_count = count
                best_user = user_id
        
        # Check consistency threshold
        ratio = best_count / total
        if ratio < self.consistency_threshold:
            logger.debug(
                f"Consistency check failed: {best_user} at {ratio:.2f} "
                f"(need {self.consistency_threshold})"
            )
            return None, 0, 0
        
        # Calculate averages
        distances = user_distances[best_user]
        avg_distance = sum(distances) / len(distances)
        avg_confidence = 1.0 - avg_distance
        
        return best_user, avg_distance, avg_confidence
    
    def _check_cooldown(self, user_id: str, current_time: float) -> bool:
        """Check if user is past cooldown period."""
        last_event = self._user_cooldowns.get(user_id)
        if last_event is None:
            return True
        
        elapsed = current_time - last_event
        if elapsed < self.cooldown_seconds:
            logger.debug(
                f"Cooldown active for {user_id}: "
                f"{self.cooldown_seconds - elapsed:.1f}s remaining"
            )
            return False
        
        return True
    
    def _cleanup_stale_tracks(self, current_time: float) -> None:
        """Remove tracks that haven't been updated recently."""
        stale_ids = []
        
        for track_id, track in self._tracks.items():
            age = current_time - track.last_seen
            if age > self.max_track_age:
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            del self._tracks[track_id]
            self.stats["tracks_expired"] += 1
    
    def reset_track(self, track_id: str) -> None:
        """Manually reset a track (e.g., when face is lost)."""
        with self._lock:
            if track_id in self._tracks:
                del self._tracks[track_id]
    
    def clear_cooldown(self, user_id: str) -> None:
        """Clear cooldown for a specific user."""
        with self._lock:
            if user_id in self._user_cooldowns:
                del self._user_cooldowns[user_id]
    
    def get_active_tracks(self) -> Dict[str, dict]:
        """Get summary of active tracks."""
        with self._lock:
            return {
                track_id: {
                    "recognitions": len(track.recognitions),
                    "validated": track.validated,
                    "emitted": track.emitted,
                    "age": time.time() - track.first_seen,
                }
                for track_id, track in self._tracks.items()
            }
    
    def get_cooldown_status(self) -> Dict[str, float]:
        """Get remaining cooldown time for each user."""
        current_time = time.time()
        with self._lock:
            return {
                user_id: max(0, self.cooldown_seconds - (current_time - last_time))
                for user_id, last_time in self._user_cooldowns.items()
            }
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._tracks.clear()
            self._user_cooldowns.clear()
            logger.info("EventValidator reset")


class EmotionEventValidator:
    """
    Validates emotion events for recognized users.
    
    Similar to EventValidator but for emotion detection.
    Only emits emotion events for users who have been identified.
    """
    
    def __init__(
        self,
        device_id: str,
        min_duration_ms: int = 500,
        confidence_threshold: float = 0.7,
        cooldown_seconds: int = 10,
    ):
        """
        Initialize emotion event validator.
        
        Args:
            device_id: Device identifier
            min_duration_ms: Minimum emotion duration to trigger event
            confidence_threshold: Minimum confidence for emotion detection
            cooldown_seconds: Cooldown between same emotion events per user
        """
        self.device_id = device_id
        self.min_duration_ms = min_duration_ms
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Tracking: user_id -> {emotion_code: (start_time, confidence_sum, count)}
        self._emotion_tracks: Dict[str, Dict[str, tuple]] = defaultdict(dict)
        self._cooldowns: Dict[str, Dict[str, float]] = defaultdict(dict)  # user -> emotion -> time
        self._lock = Lock()
    
    def process_emotion(
        self,
        user_id: str,
        emotion_code: str,
        confidence: float,
        frame_id: int,
    ) -> Optional[EmotionEvent]:
        """
        Process emotion detection for a recognized user.
        
        Args:
            user_id: Recognized user ID
            emotion_code: Detected emotion code (e.g., "EV2.2")
            confidence: Detection confidence
            frame_id: Current frame number
            
        Returns:
            EmotionEvent if validation passes, None otherwise
        """
        if confidence < self.confidence_threshold:
            return None
        
        current_time = time.time()
        current_ms = int(current_time * 1000)
        
        with self._lock:
            user_tracks = self._emotion_tracks[user_id]
            
            if emotion_code not in user_tracks:
                # Start tracking this emotion
                user_tracks[emotion_code] = (current_ms, confidence, 1)
                return None
            
            start_ms, conf_sum, count = user_tracks[emotion_code]
            duration = current_ms - start_ms
            
            # Update tracking
            user_tracks[emotion_code] = (start_ms, conf_sum + confidence, count + 1)
            
            # Check if duration threshold met
            if duration < self.min_duration_ms:
                return None
            
            # Check cooldown
            last_event = self._cooldowns[user_id].get(emotion_code, 0)
            if current_time - last_event < self.cooldown_seconds:
                return None
            
            # Emit event
            avg_confidence = (conf_sum + confidence) / (count + 1)
            
            event = EmotionEvent(
                device_id=self.device_id,
                user_id=user_id,
                emotion_code=emotion_code,
                confidence=avg_confidence,
                metadata=EventMetadata(
                    intensity=avg_confidence,
                    duration_ms=duration,
                ),
            )
            
            # Reset tracking and set cooldown
            del user_tracks[emotion_code]
            self._cooldowns[user_id][emotion_code] = current_time
            
            logger.info(
                f"Emotion event validated: user={user_id}, "
                f"emotion={emotion_code}, duration={duration}ms"
            )
            
            return event
    
    def reset_user(self, user_id: str) -> None:
        """Reset emotion tracking for a user."""
        with self._lock:
            if user_id in self._emotion_tracks:
                del self._emotion_tracks[user_id]
