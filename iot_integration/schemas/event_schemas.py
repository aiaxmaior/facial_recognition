"""
Event payload schemas for IoT broker transmission.

Defines the JSON structure for facial_id and emotion events
sent from edge devices to the IoT broker.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import uuid


class EmotionCode(str, Enum):
    """Emotion codes matching IoT_EventLookup collection."""
    NEUTRAL = "EV2.1"
    HAPPINESS = "EV2.2"
    ANGER = "EV2.3"
    CONTEMPT = "EV2.4"
    DISGUST = "EV2.5"
    FEAR = "EV2.6"
    SADNESS = "EV2.7"
    SURPRISE = "EV2.8"


class EventMetadata(BaseModel):
    """Additional metadata for events."""
    distance: Optional[float] = Field(None, ge=0, le=1, description="Cosine distance score")
    frames_tracked: Optional[int] = Field(None, ge=1, description="Number of frames tracked before validation")
    inference_ms: Optional[int] = Field(None, ge=0, description="Inference time in milliseconds")
    intensity: Optional[float] = Field(None, ge=0, le=1, description="Emotion intensity (emotion events only)")
    duration_ms: Optional[int] = Field(None, ge=0, description="Duration of emotion in milliseconds")
    face_bbox: Optional[list[int]] = Field(None, description="Face bounding box [x, y, w, h]")
    

def generate_event_id(event_type: str, emotion_code: Optional[str] = None) -> str:
    """
    Generate event ID following IoT broker conventions.
    
    Format:
        - Face recognition: EV1-{timestamp}-{random}
        - Emotion monitoring: {emotion_code}-{timestamp}-{random}
    """
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    random_suffix = uuid.uuid4().hex[:8]
    
    if event_type == "facial_id":
        return f"EV1-{timestamp}-{random_suffix}"
    elif event_type == "emotion" and emotion_code:
        return f"{emotion_code}-{timestamp}-{random_suffix}"
    else:
        return f"EV0-{timestamp}-{random_suffix}"


class EventPayload(BaseModel):
    """
    Base event payload for IoT broker transmission.
    
    This is the minimal data sent to the IoT broker for each validated event.
    Follows the schema defined in IOT_BROKER_FRAMEWORK_revised.md.
    """
    event_id: str = Field(default_factory=lambda: generate_event_id("facial_id"))
    device_id: str = Field(..., description="Unique device identifier (e.g., 'cam-001')")
    user_id: str = Field(..., description="Central WFM employee ID")
    event_type: Literal["facial_id", "emotion"] = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp (UTC)")
    image: Optional[str] = Field(None, description="Base64-encoded compressed JPEG image")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }
    
    def to_broker_json(self) -> dict:
        """Convert to JSON format expected by IoT broker."""
        return {
            "event_id": self.event_id,
            "device_id": self.device_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() + "Z",
            "image": self.image,
        }


class FacialIdEvent(EventPayload):
    """
    Facial identification event payload.
    
    Sent when a face is successfully identified and validated
    across multiple consecutive frames.
    """
    event_type: Literal["facial_id"] = "facial_id"
    confidence: float = Field(..., ge=0, le=1, description="Recognition confidence (1 - distance)")
    metadata: Optional[EventMetadata] = Field(default=None, description="Additional event metadata")
    
    def __init__(self, **data):
        if "event_id" not in data:
            data["event_id"] = generate_event_id("facial_id")
        super().__init__(**data)
    
    def to_broker_json(self) -> dict:
        """Convert to JSON format expected by IoT broker."""
        base = super().to_broker_json()
        base["confidence"] = self.confidence
        if self.metadata:
            base["metadata"] = self.metadata.model_dump(exclude_none=True)
        return base


class EmotionEvent(EventPayload):
    """
    Emotion monitoring event payload.
    
    Sent when an emotion is detected and validated for a recognized user.
    """
    event_type: Literal["emotion"] = "emotion"
    emotion_code: EmotionCode = Field(..., description="Emotion classification code")
    confidence: float = Field(..., ge=0, le=1, description="Emotion detection confidence")
    metadata: Optional[EventMetadata] = Field(default=None, description="Additional event metadata")
    
    def __init__(self, **data):
        if "event_id" not in data:
            emotion_code = data.get("emotion_code")
            if isinstance(emotion_code, EmotionCode):
                emotion_code = emotion_code.value
            data["event_id"] = generate_event_id("emotion", emotion_code)
        super().__init__(**data)
    
    def to_broker_json(self) -> dict:
        """Convert to JSON format expected by IoT broker."""
        base = super().to_broker_json()
        base["emotion_code"] = self.emotion_code.value if isinstance(self.emotion_code, EmotionCode) else self.emotion_code
        base["confidence"] = self.confidence
        if self.metadata:
            base["metadata"] = self.metadata.model_dump(exclude_none=True)
        return base
