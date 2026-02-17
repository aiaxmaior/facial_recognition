"""
Socket.IO message schemas for IoT broker communication.

All broker messages use a single Socket.IO event 'message' with payload
structured as a two-element array: [header, data].

Routing is done via header.command_id:
    Edge -> Broker:
        device.register   - register device on connect
    Broker -> Edge:
        response.success   - registration confirmed
        enrollment.publish - push a face embedding to the device
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class BrokerMessageHeader(BaseModel):
    """
    Header portion of every Socket.IO [header, data] message.

    Same structure used by both inbound and outbound messages.
    Routing is determined by command_id.
    """
    to: str = Field("gateway", description="Message destination")
    from_field: str = Field(..., alias="from", description="Sender identifier")
    source_type: str = Field("device", description="'device' or 'server'")
    auth_token: Optional[str] = Field(None, description="JWT or API key")
    command_id: str = Field(..., description="Message type / routing key")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    class Config:
        populate_by_name = True


class DeviceRegisterData(BaseModel):
    """Data payload for device.register (edge -> broker)."""
    request_id: str = Field(..., description="Unique request identifier")
    device_category: str = Field("camera", description="Device category")
    capability: str = Field("face_recognition", description="Device capability")


class EnrollmentPublishData(BaseModel):
    """
    Data payload for enrollment.publish (broker -> edge).

    The device only stores employee_id + embedded_file.
    Other fields are accepted but not persisted.
    """
    employee_id: str = Field(..., description="Central employee identifier")
    embedded_file: str = Field(..., description="Base64-encoded float32 embedding")
    person_id: Optional[str] = Field(None, description="Person ID (ignored)")
    person_name: Optional[str] = Field(None, description="Display name (ignored)")
    enrollment_data: Optional[Any] = Field(None, description="Extra enrollment data (ignored)")
