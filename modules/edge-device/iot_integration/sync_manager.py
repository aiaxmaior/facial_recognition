"""
Sync Manager for Enrollment Database Synchronization.

Handles enrollment.publish messages pushed from the IoT broker via
Socket.IO and applies them to the local SQLite database.

Push-based flow:
    1. Broker pushes enrollment.publish via Socket.IO
    2. SyncManager decodes the embedded_file and upserts into SQLite
    3. Fires on_sync_complete callback so the pipeline hot-reloads

Only employee_id and the embedding are persisted.
"""

import logging
from datetime import datetime
from typing import Callable

from .db_manager import DatabaseManager
from .image_utils import decode_embedding_b64

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Processes enrollment push messages and applies them to the local database.

    This is a stateless message handler -- no background threads. The Socket.IO
    client calls handle_enrollment_update(data_dict) when a message arrives.

    Usage:
        db = DatabaseManager("faces.db")
        sync_mgr = SyncManager(db, on_sync_complete=callback)

        # Called by WebSocketClient when enrollment.publish arrives:
        sync_mgr.handle_enrollment_update({"employee_id": "...", "embedded_file": "..."})
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        model: str = "ArcFace",
        on_sync_complete: Callable[[int, int, int], None] = None,
    ):
        """
        Initialize sync manager.

        Args:
            db_manager: Database manager instance
            model: Embedding model name (for DB record)
            on_sync_complete: Callback(additions, removals, new_version) on change
        """
        self.db = db_manager
        self.model = model
        self.on_sync_complete = on_sync_complete

        # Statistics
        self.stats = {
            "updates_applied": 0,
            "updates_failed": 0,
            "last_update_at": None,
        }

    def handle_enrollment_update(self, data: dict) -> bool:
        """
        Process an enrollment.publish message from the broker.

        Extracts employee_id and embedded_file, decodes the embedding,
        upserts into SQLite, and fires the callback to hot-reload the
        recognition pipeline.

        Args:
            data: The data dict from the broker [header, data] message.
                  Must contain 'employee_id' and 'embedded_file'.

        Returns:
            True if the enrollment was successfully applied
        """
        employee_id = data.get("employee_id")
        embedded_file = data.get("embedded_file")

        if not employee_id or not embedded_file:
            logger.error(
                f"enrollment.publish missing required fields: "
                f"employee_id={bool(employee_id)}, embedded_file={bool(embedded_file)}"
            )
            self.stats["updates_failed"] += 1
            return False

        try:
            embedding = decode_embedding_b64(embedded_file)

            success = self.db.upsert_enrollment(
                user_id=employee_id,
                embedding=embedding,
                model=self.model,
            )

            if not success:
                logger.error(f"DB upsert failed for {employee_id}")
                self.stats["updates_failed"] += 1
                return False

            self.stats["updates_applied"] += 1
            self.stats["last_update_at"] = datetime.utcnow().isoformat()

            logger.info(
                f"Enrollment applied: employee_id={employee_id}, "
                f"embedding_size={len(embedding)}"
            )

            if self.on_sync_complete:
                self.on_sync_complete(1, 0, 0)

            return True

        except Exception as e:
            logger.error(f"Failed to apply enrollment {employee_id}: {e}")
            self.stats["updates_failed"] += 1
            return False

    def get_sync_status(self) -> dict:
        """Get current sync status."""
        return {
            "enrollment_count": self.db.get_enrollment_count(),
            "stats": self.stats,
        }
