"""
Pydantic schemas for IoT integration.

This module defines all data models used for:
- Event payloads (facial_id, emotion)
- WebSocket message schemas (Socket.IO broker protocol)
- Database records
- Device configuration
"""

from .event_schemas import (
    EventPayload,
    FacialIdEvent,
    EmotionEvent,
    EventMetadata,
)
from .ws_schemas import (
    BrokerMessageHeader,
    DeviceRegisterData,
    EnrollmentPublishData,
)
from .db_schemas import (
    EnrollmentRecord,
    DeviceConfig,
    LocalEventRecord,
)

__all__ = [
    # Event schemas
    "EventPayload",
    "FacialIdEvent",
    "EmotionEvent",
    "EventMetadata",
    # WebSocket schemas
    "BrokerMessageHeader",
    "DeviceRegisterData",
    "EnrollmentPublishData",
    # DB schemas
    "EnrollmentRecord",
    "DeviceConfig",
    "LocalEventRecord",
]
