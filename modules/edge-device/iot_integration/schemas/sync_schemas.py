"""
Sync protocol schemas for enrollment database synchronization.

Defines request/response models for the PULL sync between
edge devices and the IoT broker/central dashboard.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class EnrollmentAddition(BaseModel):
    """
    Single enrollment record to add/update on device.
    
    Contains the minimal data needed for local face matching:
    - user_id: The central WFM employee ID
    - embedding: Base64-encoded float32 array
    - dim: Embedding dimension (e.g., 512 for ArcFace)
    - model: Model name for compatibility checking
    """
    user_id: str = Field(..., description="Central WFM employee ID")
    embedding: str = Field(..., description="Base64-encoded float32 embedding array")
    dim: int = Field(..., ge=128, le=2048, description="Embedding dimension")
    model: str = Field(..., description="Model name (e.g., 'ArcFace')")
    detector: Optional[str] = Field(None, description="Detector used during enrollment")
    display_name: Optional[str] = Field(None, description="Display name (dev mode only)")


class EnrollmentRemoval(BaseModel):
    """
    Enrollment removal notification.
    
    Indicates a user_id should be removed from the local database.
    """
    user_id: str = Field(..., description="User ID to remove")
    removed_at: datetime = Field(default_factory=datetime.utcnow)


class SyncRequest(BaseModel):
    """
    Request payload for enrollment sync.
    
    Sent by edge device to broker to request enrollment updates.
    """
    device_id: str = Field(..., description="Device requesting sync")
    since_version: int = Field(0, ge=0, description="Last known sync version")
    current_count: Optional[int] = Field(None, description="Current local enrollment count")
    model: str = Field("ArcFace", description="Expected embedding model")
    force_full_sync: bool = Field(False, description="Force full database sync")


class SyncResponse(BaseModel):
    """
    Response payload for enrollment sync.
    
    Contains incremental updates since the requested version.
    """
    sync_version: int = Field(..., description="Current sync version")
    additions: list[EnrollmentAddition] = Field(default_factory=list, description="New/updated enrollments")
    removals: list[str] = Field(default_factory=list, description="User IDs to remove")
    full_sync_required: bool = Field(False, description="True if device needs full re-sync")
    total_enrolled: int = Field(..., description="Total enrollments in central database")
    synced_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class SyncStatus(BaseModel):
    """
    Local sync status tracking.
    
    Stored in device_config table to track sync state.
    """
    last_sync_version: int = Field(0, description="Last successfully synced version")
    last_sync_at: Optional[datetime] = Field(None, description="Timestamp of last sync")
    local_count: int = Field(0, description="Local enrollment count")
    pending_sync: bool = Field(False, description="True if sync is needed")
    sync_errors: int = Field(0, description="Consecutive sync error count")
    last_error: Optional[str] = Field(None, description="Last sync error message")
