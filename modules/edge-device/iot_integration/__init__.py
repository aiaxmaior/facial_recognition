"""
IoT Integration Module for Facial Recognition System

This module provides integration with IoT brokers and central WFM dashboards
for facial recognition and emotion monitoring edge devices.

Components:
    - schemas: Pydantic models for events, sync, and database schemas
    - event_validator: Temporal tracking and event validation
    - iot_client: Device-side IoT broker communication
    - sync_manager: SQLite ↔ MongoDB enrollment sync
    - db_manager: Local SQLite database management
    - image_utils: Image compression for event payloads
"""

from .schemas import (
    EventPayload,
    FacialIdEvent,
    EmotionEvent,
    SyncRequest,
    SyncResponse,
    EnrollmentRecord,
    DeviceConfig,
)
from .event_validator import EventValidator
from .iot_client import IoTClient, IoTClientConfig
from .sync_manager import SyncManager
from .db_manager import DatabaseManager
from .image_utils import compress_image_for_event
from .adapter import IoTAdapter, LiveStreamWithIoT

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "EventPayload",
    "FacialIdEvent", 
    "EmotionEvent",
    "SyncRequest",
    "SyncResponse",
    "EnrollmentRecord",
    "DeviceConfig",
    # Core components
    "EventValidator",
    "IoTClient",
    "IoTClientConfig",
    "SyncManager",
    "DatabaseManager",
    "compress_image_for_event",
    # Adapter for existing system
    "IoTAdapter",
    "LiveStreamWithIoT",
]
