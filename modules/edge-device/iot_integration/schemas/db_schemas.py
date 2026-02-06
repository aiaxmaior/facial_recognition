"""
Database schemas for local SQLite storage.

Defines the restructured schema for edge device databases,
using user_id as the primary identifier instead of name.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import numpy as np


# =============================================================================
# SQLite Table Definitions (as SQL strings)
# =============================================================================

# enrolled_users: face embeddings for recognition.
# detector + model must match the recognition pipeline (e.g. yolov8n-face + ArcFace).
# Do not reuse embeddings from a different detector (e.g. retinaface) when switching to yolov8n-face.
ENROLLED_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS enrolled_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    display_name TEXT,
    model TEXT NOT NULL,
    detector TEXT,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    sync_version INTEGER DEFAULT 0,
    synced_at TEXT,
    created_at TEXT NOT NULL
);
"""

ENROLLED_USERS_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_enrolled_user_id ON enrolled_users(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_enrolled_sync_version ON enrolled_users(sync_version);",
]

DEVICE_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS device_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

LOCAL_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS local_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    device_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    emotion_code TEXT,
    confidence REAL,
    distance REAL,
    timestamp TEXT NOT NULL,
    image_path TEXT,
    metadata TEXT,
    transmitted INTEGER DEFAULT 0,
    transmitted_at TEXT,
    created_at TEXT NOT NULL
);
"""

LOCAL_EVENTS_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON local_events(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_events_user_id ON local_events(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_transmitted ON local_events(transmitted);",
]

ARCHIVE_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS archive_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    device_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    emotion_code TEXT,
    confidence REAL,
    distance REAL,
    timestamp TEXT NOT NULL,
    image_path TEXT,
    video_clip_path TEXT,
    metadata TEXT,
    frames_count INTEGER,
    duration_ms INTEGER,
    created_at TEXT NOT NULL
);
"""


# =============================================================================
# Pydantic Models for DB Records
# =============================================================================

class EnrollmentRecord(BaseModel):
    """
    Pydantic model for enrolled_users table records.
    
    Used for validation and serialization when interacting with SQLite.
    """
    id: Optional[int] = None
    user_id: str = Field(..., description="Central WFM employee ID")
    display_name: Optional[str] = Field(None, description="Display name (dev mode only)")
    model: str = Field(..., description="Embedding model name")
    detector: Optional[str] = Field(None, description="Face detector used")
    embedding: bytes = Field(..., description="Float32 embedding as bytes")
    embedding_dim: int = Field(..., description="Embedding dimension")
    sync_version: int = Field(0, description="Sync version when this record was updated")
    synced_at: Optional[datetime] = Field(None, description="When this record was synced")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_embedding_array(self) -> np.ndarray:
        """Convert embedding bytes to numpy array."""
        return np.frombuffer(self.embedding, dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, user_id: str, embedding: np.ndarray, model: str, 
                   detector: str = None, display_name: str = None,
                   sync_version: int = 0) -> "EnrollmentRecord":
        """Create record from numpy embedding array."""
        return cls(
            user_id=user_id,
            display_name=display_name,
            model=model,
            detector=detector,
            embedding=embedding.astype(np.float32).tobytes(),
            embedding_dim=len(embedding),
            sync_version=sync_version,
        )


class DeviceConfig(BaseModel):
    """
    Device configuration model.
    
    Represents key-value pairs stored in device_config table.
    """
    device_id: str = Field(..., description="Unique device identifier")
    broker_url: str = Field(..., description="IoT broker URL")
    sync_interval_minutes: int = Field(15, description="Enrollment sync interval")
    last_sync_version: int = Field(0, description="Last synced version")
    last_sync_at: Optional[datetime] = Field(None)
    dev_mode: bool = Field(False, description="Development mode flag")
    
    # Event validation settings
    confirmation_frames: int = Field(5, description="Frames needed to confirm identity")
    consistency_threshold: float = Field(0.8, description="Required consistency ratio")
    cooldown_seconds: int = Field(30, description="Cooldown between same-user events")
    
    # Image compression settings
    image_quality: int = Field(65, description="JPEG quality for event images")
    image_max_size_kb: int = Field(50, description="Max image size in KB")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "device_id": self.device_id,
            "broker_url": self.broker_url,
            "sync_interval_minutes": str(self.sync_interval_minutes),
            "last_sync_version": str(self.last_sync_version),
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else "",
            "dev_mode": "1" if self.dev_mode else "0",
            "confirmation_frames": str(self.confirmation_frames),
            "consistency_threshold": str(self.consistency_threshold),
            "cooldown_seconds": str(self.cooldown_seconds),
            "image_quality": str(self.image_quality),
            "image_max_size_kb": str(self.image_max_size_kb),
        }


class LocalEventRecord(BaseModel):
    """
    Pydantic model for local_events table records.
    
    Used for local event storage and archive before/after transmission.
    """
    id: Optional[int] = None
    event_id: str = Field(..., description="Unique event ID")
    device_id: str = Field(..., description="Device that generated the event")
    user_id: str = Field(..., description="Identified user")
    event_type: str = Field(..., description="Event type (facial_id, emotion)")
    emotion_code: Optional[str] = Field(None, description="Emotion code if applicable")
    confidence: Optional[float] = Field(None, description="Recognition/emotion confidence")
    distance: Optional[float] = Field(None, description="Cosine distance")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_path: Optional[str] = Field(None, description="Path to stored event image")
    metadata: Optional[str] = Field(None, description="JSON metadata string")
    transmitted: bool = Field(False, description="Whether event was sent to broker")
    transmitted_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
