"""
IoT Client for Edge Device Communication.

Handles all communication with the IoT broker:
- Event transmission (facial_id, emotion)
- Enrollment sync requests
- Heartbeat/status reporting
"""

import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .schemas.event_schemas import EventPayload, FacialIdEvent, EmotionEvent
from .schemas.sync_schemas import SyncRequest, SyncResponse
from .image_utils import compress_image_for_event

logger = logging.getLogger(__name__)


@dataclass
class IoTClientConfig:
    """Configuration for IoT client."""
    device_id: str
    broker_url: str
    api_key: Optional[str] = None
    
    # Retry settings
    max_retries: int = 3
    retry_backoff: float = 0.5
    
    # Batching settings
    batch_size: int = 10
    batch_timeout_ms: int = 5000
    
    # Connection settings
    timeout_seconds: int = 30
    verify_ssl: bool = True
    
    # Event queue settings
    max_queue_size: int = 1000
    
    # Image settings
    compress_images: bool = True
    image_quality: int = 65
    image_max_size_kb: int = 50


class IoTClient:
    """
    Client for communicating with the IoT broker.
    
    Features:
    - Async event queue with background transmission
    - Automatic retry with exponential backoff
    - Event batching for efficiency
    - Offline queue persistence
    
    Usage:
        client = IoTClient(config)
        client.start()
        
        # Send events
        client.send_event(facial_event)
        
        # Stop cleanly
        client.stop()
    """
    
    def __init__(self, config: IoTClientConfig):
        """
        Initialize IoT client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self._event_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._session: Optional[requests.Session] = None
        
        # Callbacks
        self.on_event_sent: Optional[Callable[[str], None]] = None
        self.on_event_failed: Optional[Callable[[str, str], None]] = None
        self.on_sync_complete: Optional[Callable[[SyncResponse], None]] = None
        
        # Statistics
        self.stats = {
            "events_queued": 0,
            "events_sent": 0,
            "events_failed": 0,
            "video_clips_sent": 0,
            "video_clips_failed": 0,
            "sync_requests": 0,
            "sync_failures": 0,
            "bytes_sent": 0,
        }
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "Content-Type": "application/json",
            "X-Device-ID": self.config.device_id,
        })
        
        if self.config.api_key:
            session.headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return session
    
    def start(self) -> None:
        """Start the IoT client background worker."""
        if self._running:
            return
        
        self._session = self._create_session()
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._event_worker,
            daemon=True,
            name="IoTClient-Worker"
        )
        self._worker_thread.start()
        logger.info(f"IoT client started for device: {self.config.device_id}")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the IoT client.
        
        Args:
            timeout: Seconds to wait for queue to drain
        """
        if not self._running:
            return
        
        self._running = False
        
        # Wait for worker to finish
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        
        # Close session
        if self._session:
            self._session.close()
            self._session = None
        
        logger.info(
            f"IoT client stopped. Events sent: {self.stats['events_sent']}, "
            f"failed: {self.stats['events_failed']}"
        )
    
    def send_event(
        self,
        event: EventPayload,
        image: "np.ndarray" = None,
        face_bbox: List[int] = None,
        blocking: bool = False,
    ) -> bool:
        """
        Queue an event for transmission.
        
        Args:
            event: Event payload to send
            image: Optional image to attach (will be compressed)
            face_bbox: Face bounding box for image cropping
            blocking: If True, wait for transmission
            
        Returns:
            True if queued successfully
        """
        # Compress and attach image if provided
        if image is not None and self.config.compress_images:
            event.image = compress_image_for_event(
                image,
                target_size_kb=self.config.image_max_size_kb,
                initial_quality=self.config.image_quality,
                face_bbox=face_bbox,
            )
        
        try:
            if blocking:
                self._event_queue.put(event, block=True, timeout=5.0)
            else:
                self._event_queue.put_nowait(event)
            
            self.stats["events_queued"] += 1
            logger.debug(f"Event queued: {event.event_id}")
            return True
            
        except queue.Full:
            logger.error("Event queue full - dropping event")
            self.stats["events_failed"] += 1
            return False
    
    def _event_worker(self) -> None:
        """Background worker for event transmission."""
        batch: List[EventPayload] = []
        batch_start_time = time.time()
        batch_timeout = self.config.batch_timeout_ms / 1000.0
        
        while self._running or not self._event_queue.empty():
            try:
                # Get event from queue with timeout
                try:
                    event = self._event_queue.get(timeout=0.1)
                    batch.append(event)
                except queue.Empty:
                    pass
                
                # Check if we should send batch
                should_send = (
                    len(batch) >= self.config.batch_size or
                    (len(batch) > 0 and time.time() - batch_start_time >= batch_timeout)
                )
                
                if should_send:
                    self._send_batch(batch)
                    batch = []
                    batch_start_time = time.time()
                    
            except Exception as e:
                logger.error(f"Event worker error: {e}")
        
        # Send remaining events on shutdown
        if batch:
            self._send_batch(batch)
    
    def _send_batch(self, events: List[EventPayload]) -> None:
        """Send a batch of events to the broker."""
        if not events:
            return
        
        url = f"{self.config.broker_url}/data/events"
        
        # Build payload with Socket.IO message format
        # Each event is [header, data] structure
        messages = []
        for e in events:
            # Use new to_broker_message() if available, fallback to legacy
            if hasattr(e, 'to_broker_message'):
                messages.append(e.to_broker_message())
            else:
                # Legacy format - wrap in header/data structure
                messages.append([
                    {"header": {
                        "to": "gateway",
                        "from": self.config.device_id,
                        "source_type": "device",
                        "auth_token": self.config.api_key,
                        "command_id": "event.log",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }},
                    {"data": e.to_broker_json()}
                ])
        
        payload = {
            "device_id": self.config.device_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "messages": messages,
        }
        
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl,
            )
            
            response.raise_for_status()
            
            # Update stats
            self.stats["events_sent"] += len(events)
            self.stats["bytes_sent"] += len(json.dumps(payload))
            
            # Callbacks
            for event in events:
                if self.on_event_sent:
                    self.on_event_sent(event.event_id)
            
            logger.debug(f"Sent batch of {len(events)} events")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send event batch: {e}")
            self.stats["events_failed"] += len(events)
            
            # Callbacks
            for event in events:
                if self.on_event_failed:
                    self.on_event_failed(event.event_id, str(e))
    
    def send_event_sync(
        self,
        event: EventPayload,
        image: "np.ndarray" = None,
        face_bbox: List[int] = None,
    ) -> bool:
        """
        Send a single event synchronously (blocking).
        
        Args:
            event: Event payload to send
            image: Optional image to attach
            face_bbox: Face bounding box for cropping
            
        Returns:
            True if sent successfully
        """
        # Compress and attach image
        if image is not None and self.config.compress_images:
            event.image = compress_image_for_event(
                image,
                target_size_kb=self.config.image_max_size_kb,
                initial_quality=self.config.image_quality,
                face_bbox=face_bbox,
            )
        
        url = f"{self.config.broker_url}/data/events"
        
        # Build Socket.IO message format
        if hasattr(event, 'to_broker_message'):
            message = event.to_broker_message()
        else:
            # Legacy format
            message = [
                {"header": {
                    "to": "gateway",
                    "from": self.config.device_id,
                    "source_type": "device",
                    "auth_token": self.config.api_key,
                    "command_id": "event.log",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }},
                {"data": event.to_broker_json()}
            ]
        
        payload = {
            "device_id": self.config.device_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "messages": [message],
        }
        
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            
            self.stats["events_sent"] += 1
            logger.info(f"Event sent: {event.event_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send event: {e}")
            self.stats["events_failed"] += 1
            return False
    
    # =========================================================================
    # Sync Operations
    # =========================================================================
    
    def request_sync(
        self,
        since_version: int = 0,
        force_full: bool = False,
        current_count: int = None,
        model: str = "ArcFace",
    ) -> Optional[SyncResponse]:
        """
        Request enrollment sync from broker.
        
        Args:
            since_version: Only get updates after this version
            force_full: Force full database sync
            current_count: Current local enrollment count
            model: Expected embedding model
            
        Returns:
            SyncResponse with updates, or None on failure
        """
        url = f"{self.config.broker_url}/data/enrollments/sync"
        
        request = SyncRequest(
            device_id=self.config.device_id,
            since_version=since_version,
            current_count=current_count,
            model=model,
            force_full_sync=force_full,
        )
        
        try:
            self.stats["sync_requests"] += 1
            
            response = self._session.get(
                url,
                params={
                    "device_id": request.device_id,
                    "since_version": request.since_version,
                    "force_full": "true" if request.force_full_sync else "false",
                    "model": request.model,
                },
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            
            data = response.json()
            sync_response = SyncResponse(**data)
            
            logger.info(
                f"Sync response: version={sync_response.sync_version}, "
                f"additions={len(sync_response.additions)}, "
                f"removals={len(sync_response.removals)}"
            )
            
            if self.on_sync_complete:
                self.on_sync_complete(sync_response)
            
            return sync_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Sync request failed: {e}")
            self.stats["sync_failures"] += 1
            return None
        except Exception as e:
            logger.error(f"Failed to parse sync response: {e}")
            self.stats["sync_failures"] += 1
            return None
    
    # =========================================================================
    # Video Clip Upload
    # =========================================================================
    
    def send_video_clip(
        self,
        event_id: str,
        video_b64: str,
        story: Optional[str] = None,
        debug: Optional[List] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Send a video clip for an event to the IoT broker.
        
        The video clip is base64 encoded and sent as part of a JSON payload
        using the Socket.IO message format (header + data).
        
        Args:
            event_id: Event identifier (format: {device_id}_{event_id})
            video_b64: Base64-encoded MP4 video data
            story: Optional VLM narrative description (~100 tokens)
            debug: Graylog debug entries (defaults to empty array)
            metadata: Optional additional metadata
            
        Returns:
            True if sent successfully
        """
        url = f"{self.config.broker_url}/data/events/video"
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Build Socket.IO message format
        message = [
            {"header": {
                "to": "gateway",
                "from": self.config.device_id,
                "source_type": "device",
                "auth_token": self.config.api_key,
                "command_id": "event.log",
                "timestamp": timestamp
            }},
            {"data": {
                "event_id": event_id,
                "event_type": "emotion_monitoring",
                "video_clip": video_b64,
                "story": story,
                "metadata": metadata or {},
                "debug": debug if debug is not None else []  # REQUIRED field
            }}
        ]
        
        payload = {
            "device_id": self.config.device_id,
            "timestamp": timestamp,
            "messages": [message],
        }
        
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds * 2,  # Allow more time for video
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            
            self.stats["video_clips_sent"] += 1
            self.stats["bytes_sent"] += len(video_b64)
            logger.info(f"Video clip sent for event: {event_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send video clip: {e}")
            self.stats["video_clips_failed"] += 1
            return False
    
    # =========================================================================
    # Heartbeat
    # =========================================================================
    
    def send_heartbeat(
        self,
        metrics: Dict = None,
        cv_stats: Dict = None,
    ) -> bool:
        """
        Send heartbeat to broker.
        
        Args:
            metrics: System metrics (cpu, memory, etc.)
            cv_stats: CV processing statistics
            
        Returns:
            True if successful
        """
        url = f"{self.config.broker_url}/data/devices/{self.config.device_id}/heartbeat"
        
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "operational",
            "metrics": metrics or {},
            "cv_stats": cv_stats or {},
            "queue_depth": self._event_queue.qsize(),
        }
        
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=10,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check if config update is needed
            if data.get("config_update_available"):
                logger.info("Configuration update available from broker")
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def get_queue_size(self) -> int:
        """Get current event queue size."""
        return self._event_queue.qsize()
    
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._running
    
    def get_stats(self) -> Dict:
        """Get client statistics."""
        return {
            **self.stats,
            "queue_size": self._event_queue.qsize(),
            "running": self._running,
        }
