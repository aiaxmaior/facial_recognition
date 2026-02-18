"""
IoT Integration Module for Facial Recognition System

Components:
    - schemas: Pydantic models for events, WebSocket messages, and database
    - event_validator: Temporal tracking and event validation
    - iot_client: HTTP outbound (events, heartbeats, video clips)
    - ws_client: Socket.IO inbound (enrollment push from broker)
    - sync_manager: Enrollment push handler (broker -> SQLite -> pipeline)
    - db_manager: Local SQLite database management
    - image_utils: Image compression for event payloads
"""

from .schemas import (
    EventPayload,
    FacialIdEvent,
    EmotionEvent,
    BrokerMessageHeader,
    EnrollmentPublishData,
    EnrollmentRecord,
    DeviceConfig,
)
from .event_validator import EventValidator
from .iot_client import IoTClient, IoTClientConfig
from .ws_client import WebSocketClient, WebSocketConfig
from .sync_manager import SyncManager
from .db_manager import DatabaseManager, EnrollmentDBManager
from .image_utils import compress_image_for_event

__version__ = "2.0.0"
__all__ = [
    # Schemas
    "EventPayload",
    "FacialIdEvent",
    "EmotionEvent",
    "BrokerMessageHeader",
    "EnrollmentPublishData",
    "EnrollmentRecord",
    "DeviceConfig",
    # Core components
    "EventValidator",
    "IoTClient",
    "IoTClientConfig",
    "WebSocketClient",
    "WebSocketConfig",
    "SyncManager",
    "DatabaseManager",
    "EnrollmentDBManager",
    "compress_image_for_event",
]
