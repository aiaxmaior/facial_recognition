"""
Event payload schemas for IoT broker transmission.

Defines the JSON structure for face_recognition and emotion_monitoring events
sent from edge devices to the IoT broker via Socket.IO.

Payload Format (Socket.IO Message):
[
  { header: { to, from, source_type, auth_token, command_id, timestamp } },
  { data: { event_id, event_type, person_name, person_id, metadata, debug, ... } }
]
"""

from datetime import datetime
from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# Event ID Generation
# =============================================================================

def generate_event_id(event_type: str, device_id: str = "device") -> str:
    """
    Generate event ID following IoT broker conventions.
    
    Format:
        - Face recognition: EV1-{timestamp}-{device_id_suffix}
        - Emotion monitoring: EV2-{timestamp}-{device_id_suffix}
    """
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    device_suffix = device_id.replace("-", "")[-8:] if device_id else uuid.uuid4().hex[:8]
    
    if event_type == "face_recognition":
        return f"EV1-{timestamp}-{device_suffix}"
    elif event_type == "emotion_monitoring":
        return f"EV2-{timestamp}-{device_suffix}"
    else:
        return f"EV0-{timestamp}-{device_suffix}"


# =============================================================================
# Message Header
# =============================================================================

class MessageHeader(BaseModel):
    """
    Socket.IO message header for IoT broker communication.
    """
    to: str = Field("gateway", description="Message destination")
    from_device: str = Field(..., alias="from", description="Device ID sending the message")
    source_type: str = Field("device", description="Source type identifier")
    auth_token: Optional[str] = Field(None, description="JWT or API key for authentication")
    command_id: str = Field("event.log", description="Command identifier")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    class Config:
        populate_by_name = True


# =============================================================================
# Event Metadata
# =============================================================================

class FaceRecognitionMetadata(BaseModel):
    """Metadata for face recognition events."""
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Recognition confidence score")
    person_detected: bool = Field(True, description="Whether a person was detected")
    distance: Optional[float] = Field(None, ge=0, le=1, description="Cosine distance score")
    frames_tracked: Optional[int] = Field(None, ge=1, description="Frames tracked before validation")
    inference_ms: Optional[int] = Field(None, ge=0, description="Inference time in milliseconds")
    face_bbox: Optional[List[int]] = Field(None, description="Face bounding box [x, y, w, h]")


class EmotionMonitoringMetadata(BaseModel):
    """Metadata for emotion monitoring events."""
    duration: Optional[float] = Field(None, ge=0, description="Video duration in seconds")
    frame_count: Optional[int] = Field(None, ge=1, description="Number of frames analyzed")
    fps: Optional[int] = Field(None, ge=1, description="Frames per second")
    resolution: Optional[str] = Field(None, description="Video resolution (e.g., '1920x1080')")
    inference_ms: Optional[int] = Field(None, ge=0, description="Inference time in milliseconds")


# =============================================================================
# Event Data Payloads
# =============================================================================

class FaceRecognitionData(BaseModel):
    """
    Data payload for face recognition events.
    
    Sent when a face is successfully identified and validated
    across multiple consecutive frames.
    """
    event_id: Optional[str] = Field(None, description="Event ID (generated if not provided)")
    event_type: Literal["face_recognition"] = "face_recognition"
    person_name: str = Field(..., min_length=1, description="Person's display name (REQUIRED, cannot be empty)")
    person_id: str = Field(..., min_length=1, description="Employee/Person ID (REQUIRED)")
    metadata: Optional[FaceRecognitionMetadata] = Field(default=None, description="Additional metadata")
    debug: List[Any] = Field(default_factory=list, description="Graylog debug entries (REQUIRED, empty array by default)")
    
    def __init__(self, **data):
        # Generate event_id if not provided
        if not data.get("event_id"):
            device_id = data.get("_device_id", "device")
            data["event_id"] = generate_event_id("face_recognition", device_id)
        # Remove internal field
        data.pop("_device_id", None)
        super().__init__(**data)


class EmotionMonitoringData(BaseModel):
    """
    Data payload for emotion monitoring events.
    
    Sent when emotional state analysis is triggered and processed by VLM.
    The 'story' field contains the VLM's narrative description (~100 tokens).
    
    Note: Raw emotion scores are stored locally on the edge device.
    The trigger criteria for emotion events (valence scores, thresholds, etc.)
    are determined by the edge device based on analysis rules.
    """
    event_id: Optional[str] = Field(None, description="Event ID (generated if not provided)")
    event_type: Literal["emotion_monitoring"] = "emotion_monitoring"
    person_id: Optional[str] = Field(None, description="Employee/Person ID if identified")
    story: Optional[str] = Field(None, description="VLM narrative description (~100 tokens)")
    video_clip: Optional[str] = Field(None, description="Base64-encoded MP4 video clip")
    metadata: Optional[EmotionMonitoringMetadata] = Field(default=None, description="Additional metadata")
    debug: List[Any] = Field(default_factory=list, description="Graylog debug entries (REQUIRED, empty array by default)")
    
    def __init__(self, **data):
        # Generate event_id if not provided
        if not data.get("event_id"):
            device_id = data.get("_device_id", "device")
            data["event_id"] = generate_event_id("emotion_monitoring", device_id)
        # Remove internal field
        data.pop("_device_id", None)
        super().__init__(**data)


# =============================================================================
# Complete Event Messages (Header + Data)
# =============================================================================

class FaceRecognitionEvent(BaseModel):
    """
    Complete face recognition event for Socket.IO transmission.
    
    Combines header and data into the expected message format.
    """
    device_id: str = Field(..., description="Device ID")
    person_name: str = Field(..., description="Person's display name")
    person_id: str = Field(..., description="Employee/Person ID")
    confidence: float = Field(..., ge=0, le=1, description="Recognition confidence")
    auth_token: Optional[str] = Field(None, description="Auth token for broker")
    metadata: Optional[FaceRecognitionMetadata] = Field(default=None)
    debug: List[Any] = Field(default_factory=list, description="Graylog entries")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: Optional[str] = Field(None)
    
    # For compatibility with existing code
    user_id: Optional[str] = Field(None, description="Alias for person_id")
    
    def __init__(self, **data):
        # Handle user_id -> person_id compatibility
        if "user_id" in data and "person_id" not in data:
            data["person_id"] = data["user_id"]
        if "person_id" in data and "user_id" not in data:
            data["user_id"] = data["person_id"]
        # Default person_name to person_id if not provided
        if "person_name" not in data and "person_id" in data:
            data["person_name"] = data["person_id"]
        super().__init__(**data)
    
    def to_broker_message(self) -> list:
        """
        Convert to Socket.IO message format expected by IoT broker.
        
        Returns:
            List containing [header_dict, data_dict]
        """
        # Generate event_id if needed
        event_id = self.event_id or generate_event_id("face_recognition", self.device_id)
        
        header = {
            "to": "gateway",
            "from": self.device_id,
            "source_type": "device",
            "auth_token": self.auth_token,
            "command_id": "event.log",
            "timestamp": self.timestamp.isoformat() + "Z"
        }
        
        data = {
            "event_id": event_id,
            "event_type": "face_recognition",
            "person_name": self.person_name,
            "person_id": self.person_id,
            "metadata": {
                "confidence": self.confidence,
                "person_detected": True,
                **(self.metadata.model_dump(exclude_none=True) if self.metadata else {})
            },
            "debug": self.debug
        }
        
        return [{"header": header}, {"data": data}]
    
    # Legacy method for backward compatibility
    def to_broker_json(self) -> dict:
        """Legacy method - returns flattened dict for older code."""
        msg = self.to_broker_message()
        return {
            **msg[0]["header"],
            **msg[1]["data"]
        }


class EmotionMonitoringEvent(BaseModel):
    """
    Complete emotion monitoring event for Socket.IO transmission.
    
    Combines header and data into the expected message format.
    The 'story' field should contain the VLM's narrative description.
    """
    device_id: str = Field(..., description="Device ID")
    person_id: Optional[str] = Field(None, description="Employee/Person ID if identified")
    story: Optional[str] = Field(None, description="VLM narrative (~100 tokens)")
    video_clip: Optional[str] = Field(None, description="Base64-encoded MP4")
    auth_token: Optional[str] = Field(None, description="Auth token for broker")
    metadata: Optional[EmotionMonitoringMetadata] = Field(default=None)
    debug: List[Any] = Field(default_factory=list, description="Graylog entries")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: Optional[str] = Field(None)
    
    def to_broker_message(self) -> list:
        """
        Convert to Socket.IO message format expected by IoT broker.
        
        Returns:
            List containing [header_dict, data_dict]
        """
        # Generate event_id if needed
        event_id = self.event_id or generate_event_id("emotion_monitoring", self.device_id)
        
        header = {
            "to": "gateway",
            "from": self.device_id,
            "source_type": "device",
            "auth_token": self.auth_token,
            "command_id": "event.log",
            "timestamp": self.timestamp.isoformat() + "Z"
        }
        
        data = {
            "event_id": event_id,
            "event_type": "emotion_monitoring",
            "story": self.story,
            "metadata": self.metadata.model_dump(exclude_none=True) if self.metadata else {},
            "debug": self.debug
        }
        
        # Add person_id if available
        if self.person_id:
            data["person_id"] = self.person_id
        
        # Add video_clip if available (base64 MP4)
        if self.video_clip:
            data["video_clip"] = self.video_clip
        
        return [{"header": header}, {"data": data}]
    
    def to_broker_json(self) -> dict:
        """Legacy method - returns flattened dict for older code."""
        msg = self.to_broker_message()
        return {
            **msg[0]["header"],
            **msg[1]["data"]
        }


# =============================================================================
# Type Aliases for Backward Compatibility
# =============================================================================

# Old names -> New names
FacialIdEvent = FaceRecognitionEvent
EmotionEvent = EmotionMonitoringEvent
EventPayload = FaceRecognitionEvent  # Default to face recognition
EventMetadata = FaceRecognitionMetadata
