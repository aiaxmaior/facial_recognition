"""
Database Manager for IoT Edge Devices.

Handles all SQLite operations for the restructured schema:
- enrolled_users: Face embeddings keyed by user_id
- device_config: Device configuration storage
- local_events: Event queue and archive

This replaces the name-based schema with user_id-based identification.
"""

import sqlite3
import threading
import logging
import json
import base64
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np

from .schemas.db_schemas import (
    EnrollmentRecord,
    DeviceConfig,
    LocalEventRecord,
    ENROLLED_USERS_TABLE_SQL,
    ENROLLED_USERS_INDEXES_SQL,
    DEVICE_CONFIG_TABLE_SQL,
    LOCAL_EVENTS_TABLE_SQL,
    LOCAL_EVENTS_INDEXES_SQL,
    ARCHIVE_EVENTS_TABLE_SQL,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for edge device storage.
    
    Manages the restructured database schema with user_id as the primary
    identifier for enrolled faces. Thread-safe for concurrent access.
    
    Usage:
        db = DatabaseManager("/path/to/database.db")
        db.initialize()
        
        # Add enrollment
        db.upsert_enrollment("EMP-123", embedding_array, "ArcFace")
        
        # Get all enrollments for matching
        enrollments = db.get_all_enrollments()
        
        # Store event
        db.store_event(event_record)
    """
    
    def __init__(self, db_path: str, dev_mode: bool = False):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
            dev_mode: If True, store display names; if False, only user_ids
        """
        self.db_path = Path(db_path)
        self.dev_mode = dev_mode
        self._lock = threading.Lock()
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize database tables and indexes.
        
        Creates all required tables if they don't exist.
        Safe to call multiple times.
        """
        with self._lock:
            if self._initialized:
                return
            
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute(ENROLLED_USERS_TABLE_SQL)
                cursor.execute(DEVICE_CONFIG_TABLE_SQL)
                cursor.execute(LOCAL_EVENTS_TABLE_SQL)
                cursor.execute(ARCHIVE_EVENTS_TABLE_SQL)
                
                # Create indexes
                for index_sql in ENROLLED_USERS_INDEXES_SQL:
                    cursor.execute(index_sql)
                for index_sql in LOCAL_EVENTS_INDEXES_SQL:
                    cursor.execute(index_sql)
                
                conn.commit()
                self._initialized = True
                logger.info(f"Database initialized: {self.db_path}")
                
            finally:
                conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    # =========================================================================
    # Enrollment Management
    # =========================================================================
    
    def upsert_enrollment(
        self,
        user_id: str,
        embedding: np.ndarray,
        model: str,
        detector: str = None,
        display_name: str = None,
        sync_version: int = 0,
    ) -> bool:
        """
        Insert or update an enrollment record.
        
        Args:
            user_id: Central WFM employee ID
            embedding: Face embedding as numpy array
            model: Embedding model name (e.g., "ArcFace")
            detector: Face detector used (optional)
            display_name: Display name (only stored in dev_mode)
            sync_version: Sync version for this record
            
        Returns:
            True if successful
        """
        # Only store display_name in dev mode
        if not self.dev_mode:
            display_name = None
        
        embedding_bytes = embedding.astype(np.float32).tobytes()
        embedding_dim = len(embedding)
        now = datetime.utcnow().isoformat()
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO enrolled_users 
                    (user_id, display_name, model, detector, embedding, embedding_dim, 
                     sync_version, synced_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        model = excluded.model,
                        detector = excluded.detector,
                        embedding = excluded.embedding,
                        embedding_dim = excluded.embedding_dim,
                        sync_version = excluded.sync_version,
                        synced_at = excluded.synced_at
                """, (
                    user_id, display_name, model, detector, 
                    embedding_bytes, embedding_dim, sync_version, now, now
                ))
                conn.commit()
                logger.debug(f"Upserted enrollment: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to upsert enrollment {user_id}: {e}")
                return False
            finally:
                conn.close()
    
    def get_enrollment(self, user_id: str) -> Optional[EnrollmentRecord]:
        """
        Get a single enrollment by user_id.
        
        Args:
            user_id: The user ID to look up
            
        Returns:
            EnrollmentRecord or None if not found
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM enrolled_users WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                if row:
                    return EnrollmentRecord(
                        id=row["id"],
                        user_id=row["user_id"],
                        display_name=row["display_name"],
                        model=row["model"],
                        detector=row["detector"],
                        embedding=row["embedding"],
                        embedding_dim=row["embedding_dim"],
                        sync_version=row["sync_version"],
                        synced_at=datetime.fromisoformat(row["synced_at"]) if row["synced_at"] else None,
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                return None
            finally:
                conn.close()
    
    def get_all_enrollments(self, model_filter: str = None) -> Dict[str, np.ndarray]:
        """
        Get all enrollments as a dictionary for matching.
        
        Args:
            model_filter: Optional model name to filter by
            
        Returns:
            Dict mapping user_id -> embedding numpy array
        """
        enrollments = {}
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if model_filter:
                    cursor.execute(
                        "SELECT user_id, embedding FROM enrolled_users WHERE model = ?",
                        (model_filter,)
                    )
                else:
                    cursor.execute("SELECT user_id, embedding FROM enrolled_users")
                
                for row in cursor.fetchall():
                    embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                    enrollments[row["user_id"]] = embedding
                
                return enrollments
            finally:
                conn.close()
    
    def get_all_enrollment_records(self) -> List[EnrollmentRecord]:
        """
        Get all enrollment records with full details.
        
        Returns:
            List of EnrollmentRecord objects
        """
        records = []
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM enrolled_users ORDER BY created_at DESC")
                
                for row in cursor.fetchall():
                    records.append(EnrollmentRecord(
                        id=row["id"],
                        user_id=row["user_id"],
                        display_name=row["display_name"],
                        model=row["model"],
                        detector=row["detector"],
                        embedding=row["embedding"],
                        embedding_dim=row["embedding_dim"],
                        sync_version=row["sync_version"],
                        synced_at=datetime.fromisoformat(row["synced_at"]) if row["synced_at"] else None,
                        created_at=datetime.fromisoformat(row["created_at"]),
                    ))
                return records
            finally:
                conn.close()
    
    def delete_enrollment(self, user_id: str) -> bool:
        """
        Delete an enrollment by user_id.
        
        Args:
            user_id: The user ID to delete
            
        Returns:
            True if a record was deleted
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM enrolled_users WHERE user_id = ?", (user_id,))
                conn.commit()
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted enrollment: {user_id}")
                return deleted
            finally:
                conn.close()
    
    def get_enrollment_count(self) -> int:
        """Get total number of enrolled users."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM enrolled_users")
                return cursor.fetchone()[0]
            finally:
                conn.close()
    
    def get_current_sync_version(self) -> int:
        """Get the maximum sync_version in the database."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(sync_version) FROM enrolled_users")
                result = cursor.fetchone()[0]
                return result if result is not None else 0
            finally:
                conn.close()
    
    # =========================================================================
    # Device Configuration
    # =========================================================================
    
    def set_config(self, key: str, value: str) -> None:
        """Set a configuration value."""
        now = datetime.utcnow().isoformat()
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO device_config (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                """, (key, value, now))
                conn.commit()
            finally:
                conn.close()
    
    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """Get a configuration value."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM device_config WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row["value"] if row else default
            finally:
                conn.close()
    
    def get_all_config(self) -> Dict[str, str]:
        """Get all configuration values as a dictionary."""
        config = {}
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM device_config")
                for row in cursor.fetchall():
                    config[row["key"]] = row["value"]
                return config
            finally:
                conn.close()
    
    def save_device_config(self, config: DeviceConfig) -> None:
        """Save a DeviceConfig object to database."""
        for key, value in config.to_dict().items():
            self.set_config(key, str(value))
    
    def load_device_config(self) -> Optional[DeviceConfig]:
        """Load DeviceConfig from database."""
        config_dict = self.get_all_config()
        if not config_dict.get("device_id"):
            return None
        
        try:
            return DeviceConfig(
                device_id=config_dict.get("device_id", ""),
                broker_url=config_dict.get("broker_url", ""),
                sync_interval_minutes=int(config_dict.get("sync_interval_minutes", "15")),
                last_sync_version=int(config_dict.get("last_sync_version", "0")),
                last_sync_at=datetime.fromisoformat(config_dict["last_sync_at"]) 
                    if config_dict.get("last_sync_at") else None,
                dev_mode=config_dict.get("dev_mode", "0") == "1",
                confirmation_frames=int(config_dict.get("confirmation_frames", "5")),
                consistency_threshold=float(config_dict.get("consistency_threshold", "0.8")),
                cooldown_seconds=int(config_dict.get("cooldown_seconds", "30")),
                image_quality=int(config_dict.get("image_quality", "65")),
                image_max_size_kb=int(config_dict.get("image_max_size_kb", "50")),
            )
        except Exception as e:
            logger.error(f"Failed to load device config: {e}")
            return None
    
    # =========================================================================
    # Event Storage
    # =========================================================================
    
    def store_event(self, event: LocalEventRecord) -> bool:
        """
        Store an event record.
        
        Args:
            event: The event record to store
            
        Returns:
            True if successful
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO local_events
                    (event_id, device_id, user_id, event_type, emotion_code,
                     confidence, distance, timestamp, image_path, metadata,
                     transmitted, transmitted_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.device_id,
                    event.user_id,
                    event.event_type,
                    event.emotion_code,
                    event.confidence,
                    event.distance,
                    event.timestamp.isoformat(),
                    event.image_path,
                    event.metadata,
                    1 if event.transmitted else 0,
                    event.transmitted_at.isoformat() if event.transmitted_at else None,
                    event.created_at.isoformat(),
                ))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                logger.warning(f"Event already exists: {event.event_id}")
                return False
            except Exception as e:
                logger.error(f"Failed to store event: {e}")
                return False
            finally:
                conn.close()
    
    def get_pending_events(self, limit: int = 100) -> List[LocalEventRecord]:
        """Get events that haven't been transmitted yet."""
        events = []
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM local_events 
                    WHERE transmitted = 0 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    events.append(LocalEventRecord(
                        id=row["id"],
                        event_id=row["event_id"],
                        device_id=row["device_id"],
                        user_id=row["user_id"],
                        event_type=row["event_type"],
                        emotion_code=row["emotion_code"],
                        confidence=row["confidence"],
                        distance=row["distance"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        image_path=row["image_path"],
                        metadata=row["metadata"],
                        transmitted=bool(row["transmitted"]),
                        transmitted_at=datetime.fromisoformat(row["transmitted_at"]) 
                            if row["transmitted_at"] else None,
                        created_at=datetime.fromisoformat(row["created_at"]),
                    ))
                return events
            finally:
                conn.close()
    
    def mark_event_transmitted(self, event_id: str) -> bool:
        """Mark an event as successfully transmitted."""
        now = datetime.utcnow().isoformat()
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE local_events 
                    SET transmitted = 1, transmitted_at = ?
                    WHERE event_id = ?
                """, (now, event_id))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
    
    def get_recent_events(
        self, 
        limit: int = 100, 
        user_id: str = None,
        event_type: str = None
    ) -> List[LocalEventRecord]:
        """Get recent events with optional filters."""
        events = []
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM local_events WHERE 1=1"
                params = []
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    events.append(LocalEventRecord(
                        id=row["id"],
                        event_id=row["event_id"],
                        device_id=row["device_id"],
                        user_id=row["user_id"],
                        event_type=row["event_type"],
                        emotion_code=row["emotion_code"],
                        confidence=row["confidence"],
                        distance=row["distance"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        image_path=row["image_path"],
                        metadata=row["metadata"],
                        transmitted=bool(row["transmitted"]),
                        transmitted_at=datetime.fromisoformat(row["transmitted_at"]) 
                            if row["transmitted_at"] else None,
                        created_at=datetime.fromisoformat(row["created_at"]),
                    ))
                return events
            finally:
                conn.close()
    
    # =========================================================================
    # Bulk Operations (for sync)
    # =========================================================================
    
    def bulk_upsert_enrollments(
        self,
        enrollments: List[Tuple[str, np.ndarray, str, str, str, int]],
    ) -> int:
        """
        Bulk insert/update enrollments.
        
        Args:
            enrollments: List of tuples (user_id, embedding, model, detector, display_name, sync_version)
            
        Returns:
            Number of records affected
        """
        now = datetime.utcnow().isoformat()
        count = 0
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                for user_id, embedding, model, detector, display_name, sync_version in enrollments:
                    if not self.dev_mode:
                        display_name = None
                    
                    embedding_bytes = embedding.astype(np.float32).tobytes()
                    embedding_dim = len(embedding)
                    
                    cursor.execute("""
                        INSERT INTO enrolled_users 
                        (user_id, display_name, model, detector, embedding, embedding_dim, 
                         sync_version, synced_at, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET
                            display_name = excluded.display_name,
                            model = excluded.model,
                            detector = excluded.detector,
                            embedding = excluded.embedding,
                            embedding_dim = excluded.embedding_dim,
                            sync_version = excluded.sync_version,
                            synced_at = excluded.synced_at
                    """, (
                        user_id, display_name, model, detector,
                        embedding_bytes, embedding_dim, sync_version, now, now
                    ))
                    count += 1
                
                conn.commit()
                logger.info(f"Bulk upserted {count} enrollments")
                return count
            except Exception as e:
                logger.error(f"Bulk upsert failed: {e}")
                conn.rollback()
                return 0
            finally:
                conn.close()
    
    def bulk_delete_enrollments(self, user_ids: List[str]) -> int:
        """
        Bulk delete enrollments by user_id.
        
        Args:
            user_ids: List of user IDs to delete
            
        Returns:
            Number of records deleted
        """
        if not user_ids:
            return 0
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(user_ids))
                cursor.execute(
                    f"DELETE FROM enrolled_users WHERE user_id IN ({placeholders})",
                    user_ids
                )
                conn.commit()
                deleted = cursor.rowcount
                logger.info(f"Bulk deleted {deleted} enrollments")
                return deleted
            finally:
                conn.close()
    
    def clear_all_enrollments(self) -> int:
        """
        Delete all enrollments (for full re-sync).
        
        Returns:
            Number of records deleted
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM enrolled_users")
                conn.commit()
                deleted = cursor.rowcount
                logger.warning(f"Cleared all {deleted} enrollments")
                return deleted
            finally:
                conn.close()
