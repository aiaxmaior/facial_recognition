"""
Sync Manager for Enrollment Database Synchronization.

Handles pulling enrollment updates from the IoT broker and
updating the local SQLite database.

Supports:
- Incremental sync (delta updates)
- Full sync (complete database refresh)
- Background periodic sync
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional, Callable
import numpy as np

from .db_manager import DatabaseManager
from .iot_client import IoTClient, IoTClientConfig
from .schemas.sync_schemas import SyncResponse, EnrollmentAddition
from .image_utils import decode_embedding_b64

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Manages enrollment database synchronization between edge device and broker.
    
    The sync flow:
    1. Device requests updates since last known version
    2. Broker returns additions/removals since that version
    3. Device applies changes to local SQLite
    4. Device updates its sync version
    
    Usage:
        db = DatabaseManager("faces.db")
        client = IoTClient(config)
        sync_mgr = SyncManager(db, client)
        
        # Start periodic sync (every 15 minutes)
        sync_mgr.start_periodic_sync(interval_minutes=15)
        
        # Or manual sync
        sync_mgr.sync_now()
        
        # Stop
        sync_mgr.stop()
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        iot_client: IoTClient,
        model: str = "ArcFace",
        on_sync_complete: Callable[[int, int, int], None] = None,
    ):
        """
        Initialize sync manager.
        
        Args:
            db_manager: Database manager instance
            iot_client: IoT client for broker communication
            model: Expected embedding model name
            on_sync_complete: Callback(additions, removals, new_version) on sync
        """
        self.db = db_manager
        self.client = iot_client
        self.model = model
        self.on_sync_complete = on_sync_complete
        
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
        self._sync_interval = 900  # 15 minutes default
        self._last_sync_time: Optional[float] = None
        self._sync_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "syncs_completed": 0,
            "syncs_failed": 0,
            "total_additions": 0,
            "total_removals": 0,
            "last_sync_at": None,
            "last_sync_version": 0,
        }
    
    def sync_now(self, force_full: bool = False) -> bool:
        """
        Perform sync immediately.
        
        Args:
            force_full: Force full database sync
            
        Returns:
            True if sync successful
        """
        with self._sync_lock:
            return self._do_sync(force_full)
    
    def _do_sync(self, force_full: bool = False) -> bool:
        """Internal sync implementation."""
        try:
            # Get current state
            current_version = self.db.get_current_sync_version()
            current_count = self.db.get_enrollment_count()
            
            logger.info(
                f"Starting sync: current_version={current_version}, "
                f"count={current_count}, force_full={force_full}"
            )
            
            # Request sync from broker
            response = self.client.request_sync(
                since_version=0 if force_full else current_version,
                force_full=force_full,
                current_count=current_count,
                model=self.model,
            )
            
            if response is None:
                logger.error("Sync request failed - no response")
                self.stats["syncs_failed"] += 1
                return False
            
            # Apply changes
            additions, removals = self._apply_sync_response(response)
            
            # Update stats
            self.stats["syncs_completed"] += 1
            self.stats["total_additions"] += additions
            self.stats["total_removals"] += removals
            self.stats["last_sync_at"] = datetime.utcnow().isoformat()
            self.stats["last_sync_version"] = response.sync_version
            self._last_sync_time = time.time()
            
            # Save sync version to config
            self.db.set_config("last_sync_version", str(response.sync_version))
            self.db.set_config("last_sync_at", datetime.utcnow().isoformat())
            
            logger.info(
                f"Sync complete: +{additions} -{removals}, "
                f"new_version={response.sync_version}"
            )
            
            # Callback
            if self.on_sync_complete:
                self.on_sync_complete(additions, removals, response.sync_version)
            
            return True
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.stats["syncs_failed"] += 1
            return False
    
    def _apply_sync_response(self, response: SyncResponse) -> tuple:
        """
        Apply sync response to local database.
        
        Args:
            response: Sync response from broker
            
        Returns:
            (additions_count, removals_count)
        """
        additions = 0
        removals = 0
        
        # Handle full sync (clear and reload)
        if response.full_sync_required:
            logger.warning("Full sync required - clearing local database")
            self.db.clear_all_enrollments()
        
        # Process additions
        if response.additions:
            enrollments = []
            
            for addition in response.additions:
                try:
                    # Decode embedding from base64
                    embedding = decode_embedding_b64(
                        addition.embedding,
                        dim=addition.dim
                    )
                    
                    enrollments.append((
                        addition.user_id,
                        embedding,
                        addition.model,
                        addition.detector,
                        addition.display_name,
                        response.sync_version,
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to decode enrollment {addition.user_id}: {e}")
            
            if enrollments:
                additions = self.db.bulk_upsert_enrollments(enrollments)
        
        # Process removals
        if response.removals:
            removals = self.db.bulk_delete_enrollments(response.removals)
        
        return additions, removals
    
    # =========================================================================
    # Periodic Sync
    # =========================================================================
    
    def start_periodic_sync(self, interval_minutes: int = 15) -> None:
        """
        Start background periodic sync.
        
        Args:
            interval_minutes: Minutes between sync attempts
        """
        if self._running:
            return
        
        self._sync_interval = interval_minutes * 60
        self._running = True
        self._sync_thread = threading.Thread(
            target=self._sync_worker,
            daemon=True,
            name="SyncManager-Worker"
        )
        self._sync_thread.start()
        
        logger.info(f"Periodic sync started: interval={interval_minutes}min")
    
    def stop(self) -> None:
        """Stop periodic sync."""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5.0)
        logger.info("Sync manager stopped")
    
    def _sync_worker(self) -> None:
        """Background worker for periodic sync."""
        # Initial sync on startup
        self._do_sync()
        
        while self._running:
            # Sleep in small increments for responsiveness
            for _ in range(int(self._sync_interval)):
                if not self._running:
                    return
                time.sleep(1)
            
            # Perform sync
            if self._running:
                self._do_sync()
    
    def trigger_sync(self) -> None:
        """Trigger an immediate sync (non-blocking)."""
        threading.Thread(
            target=self._do_sync,
            daemon=True,
            name="SyncManager-Trigger"
        ).start()
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def get_sync_status(self) -> dict:
        """Get current sync status."""
        last_version = self.db.get_config("last_sync_version", "0")
        last_sync = self.db.get_config("last_sync_at")
        
        return {
            "running": self._running,
            "interval_minutes": self._sync_interval / 60,
            "last_sync_version": int(last_version),
            "last_sync_at": last_sync,
            "enrollment_count": self.db.get_enrollment_count(),
            "stats": self.stats,
        }
    
    def needs_sync(self) -> bool:
        """Check if sync is needed based on interval."""
        if self._last_sync_time is None:
            return True
        
        elapsed = time.time() - self._last_sync_time
        return elapsed >= self._sync_interval


class SyncManagerFactory:
    """
    Factory for creating SyncManager instances.
    
    Simplifies setup by creating all dependencies.
    """
    
    @staticmethod
    def create(
        db_path: str,
        device_id: str,
        broker_url: str,
        api_key: str = None,
        model: str = "ArcFace",
        dev_mode: bool = False,
    ) -> tuple:
        """
        Create SyncManager with all dependencies.
        
        Args:
            db_path: Path to SQLite database
            device_id: Device identifier
            broker_url: IoT broker URL
            api_key: Optional API key
            model: Embedding model name
            dev_mode: Development mode flag
            
        Returns:
            (SyncManager, DatabaseManager, IoTClient)
        """
        # Create database manager
        db = DatabaseManager(db_path, dev_mode=dev_mode)
        db.initialize()
        
        # Create IoT client
        client_config = IoTClientConfig(
            device_id=device_id,
            broker_url=broker_url,
            api_key=api_key,
        )
        client = IoTClient(client_config)
        
        # Create sync manager
        sync_mgr = SyncManager(db, client, model=model)
        
        return sync_mgr, db, client
