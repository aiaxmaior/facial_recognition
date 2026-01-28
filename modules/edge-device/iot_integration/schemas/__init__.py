"""
Pydantic schemas for IoT integration.

This module defines all data models used for:
- Event payloads (facial_id, emotion)
- Sync requests/responses
- Database records
- Device configuration
"""

from .event_schemas import (
    EventPayload,
    FacialIdEvent,
    EmotionEvent,
    EventMetadata,
)
from .sync_schemas import (
    SyncRequest,
    SyncResponse,
    EnrollmentAddition,
    EnrollmentRemoval,
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
    # Sync schemas
    "SyncRequest",
    "SyncResponse",
    "EnrollmentAddition",
    "EnrollmentRemoval",
    # DB schemas
    "EnrollmentRecord",
    "DeviceConfig",
    "LocalEventRecord",
]
