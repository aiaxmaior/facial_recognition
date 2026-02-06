"""
Structured logging configuration for edge device.

Provides JSON-formatted logs suitable for Graylog ingestion and local record keeping.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs in JSON Lines format, one JSON object per line.
    Compatible with Graylog, ELK stack, and other log aggregators.
    """
    
    def __init__(
        self,
        device_id: str = "unknown",
        include_extra: bool = True,
        pretty: bool = False,
    ):
        super().__init__()
        self.device_id = device_id
        self.include_extra = include_extra
        self.pretty = pretty
        
        # Fields to exclude from extra (standard LogRecord attributes)
        self._skip_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'asctime', 'taskName'
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "device_id": self.device_id,
            "source": {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in self._skip_fields:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)
            
            if extra:
                log_entry["extra"] = extra
        
        if self.pretty:
            return json.dumps(log_entry, indent=2, default=str)
        return json.dumps(log_entry, default=str)


class EventLogger:
    """
    Dedicated logger for IoT events with structured context.
    
    Tracks event lifecycle and maintains correlation between
    client-generated and broker-assigned IDs.
    """
    
    def __init__(self, device_id: str, logger: logging.Logger = None):
        self.device_id = device_id
        self.logger = logger or logging.getLogger("iot.events")
        self._event_context: Dict[str, Dict[str, Any]] = {}
    
    def log_event_created(
        self,
        event_id: str,
        event_type: str,
        person_id: str = None,
        person_name: str = None,
        confidence: float = None,
        metadata: Dict = None,
    ) -> None:
        """Log when an event is created locally."""
        context = {
            "client_event_id": event_id,
            "event_type": event_type,
            "person_id": person_id,
            "person_name": person_name,
            "confidence": confidence,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "created",
        }
        self._event_context[event_id] = context
        
        self.logger.info(
            f"Event created: {event_id}",
            extra={
                "event_action": "created",
                "event_context": context,
                "metadata": metadata or {},
            }
        )
    
    def log_event_sent(
        self,
        event_id: str,
        endpoint: str,
        payload_size: int = None,
        has_image: bool = False,
    ) -> None:
        """Log when an event is sent to broker."""
        context = self._event_context.get(event_id, {})
        context["sent_at"] = datetime.utcnow().isoformat() + "Z"
        context["status"] = "sent"
        
        self.logger.info(
            f"Event sent: {event_id} -> {endpoint}",
            extra={
                "event_action": "sent",
                "client_event_id": event_id,
                "endpoint": endpoint,
                "payload_size_bytes": payload_size,
                "has_image": has_image,
            }
        )
    
    def log_event_acknowledged(
        self,
        client_event_id: str,
        broker_response: Dict[str, Any],
        response_time_ms: float = None,
    ) -> None:
        """Log broker acknowledgment with full response."""
        context = self._event_context.get(client_event_id, {})
        
        # Extract key fields from broker response
        broker_event_id = broker_response.get("event_id")
        broker_doc_id = broker_response.get("_id")
        device_info = broker_response.get("device", {})
        
        context["broker_event_id"] = broker_event_id
        context["broker_doc_id"] = broker_doc_id
        context["acknowledged_at"] = datetime.utcnow().isoformat() + "Z"
        context["status"] = "acknowledged"
        
        self.logger.info(
            f"Event acknowledged: {client_event_id} -> {broker_event_id}",
            extra={
                "event_action": "acknowledged",
                "client_event_id": client_event_id,
                "broker_event_id": broker_event_id,
                "broker_doc_id": broker_doc_id,
                "broker_timestamp": broker_response.get("event_timestamp"),
                "device_name": device_info.get("display_name"),
                "device_location": broker_response.get("device_location"),
                "response_time_ms": response_time_ms,
                "full_response": broker_response,
            }
        )
        
        # Update stored context
        self._event_context[client_event_id] = context
    
    def log_event_failed(
        self,
        event_id: str,
        error: str,
        status_code: int = None,
        response_body: str = None,
    ) -> None:
        """Log event transmission failure."""
        context = self._event_context.get(event_id, {})
        context["failed_at"] = datetime.utcnow().isoformat() + "Z"
        context["status"] = "failed"
        context["error"] = error
        
        self.logger.error(
            f"Event failed: {event_id} - {error}",
            extra={
                "event_action": "failed",
                "client_event_id": event_id,
                "error": error,
                "status_code": status_code,
                "response_body": response_body,
            }
        )
    
    def get_event_context(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get stored context for an event."""
        return self._event_context.get(event_id)
    
    def clear_old_contexts(self, max_age_seconds: int = 300) -> int:
        """Clear event contexts older than max_age_seconds."""
        now = datetime.utcnow()
        expired = []
        
        for event_id, context in self._event_context.items():
            created_str = context.get("created_at", "")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    age = (now - created.replace(tzinfo=None)).total_seconds()
                    if age > max_age_seconds:
                        expired.append(event_id)
                except ValueError:
                    pass
        
        for event_id in expired:
            del self._event_context[event_id]
        
        return len(expired)


def setup_logging(
    device_id: str,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    json_logs: bool = True,
) -> logging.Logger:
    """
    Configure structured logging for the edge device.
    
    Args:
        device_id: Device identifier for log context
        log_dir: Directory for log files
        console_level: Console output log level
        file_level: File output log level
        json_logs: If True, use JSON format for file logs
        
    Returns:
        Root logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler - human readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - structured JSON
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if json_logs:
        # JSON lines format for machine parsing
        json_file = log_path / f"events_{timestamp}.jsonl"
        json_handler = logging.FileHandler(json_file, encoding='utf-8')
        json_handler.setLevel(file_level)
        json_handler.setFormatter(StructuredFormatter(device_id=device_id))
        root_logger.addHandler(json_handler)
        
        # Separate event log for easy filtering
        event_logger = logging.getLogger("iot.events")
        event_file = log_path / f"iot_events_{timestamp}.jsonl"
        event_handler = logging.FileHandler(event_file, encoding='utf-8')
        event_handler.setLevel(logging.INFO)
        event_handler.setFormatter(StructuredFormatter(device_id=device_id))
        event_logger.addHandler(event_handler)
    else:
        # Plain text format
        text_file = log_path / f"device_{timestamp}.log"
        text_handler = logging.FileHandler(text_file, encoding='utf-8')
        text_handler.setLevel(file_level)
        text_handler.setFormatter(console_formatter)
        root_logger.addHandler(text_handler)
    
    return root_logger


def build_debug_entries(
    frame_id: int = None,
    detection_time_ms: float = None,
    recognition_time_ms: float = None,
    faces_detected: int = None,
    pipeline_state: str = None,
    extra: Dict[str, Any] = None,
) -> list:
    """
    Build debug entries for the event's Graylog debug field.
    
    Args:
        frame_id: Current frame number
        detection_time_ms: Face detection time in milliseconds
        recognition_time_ms: Face recognition time in milliseconds
        faces_detected: Number of faces detected
        pipeline_state: Current pipeline state
        extra: Additional debug info
        
    Returns:
        List of debug entries for the event payload
    """
    entries = []
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Timing entry
    if detection_time_ms is not None or recognition_time_ms is not None:
        timing = {
            "type": "timing",
            "timestamp": timestamp,
        }
        if detection_time_ms is not None:
            timing["detection_ms"] = round(detection_time_ms, 2)
        if recognition_time_ms is not None:
            timing["recognition_ms"] = round(recognition_time_ms, 2)
        if detection_time_ms and recognition_time_ms:
            timing["total_ms"] = round(detection_time_ms + recognition_time_ms, 2)
        entries.append(timing)
    
    # Pipeline state entry
    if frame_id is not None or faces_detected is not None or pipeline_state:
        state = {
            "type": "pipeline_state",
            "timestamp": timestamp,
        }
        if frame_id is not None:
            state["frame_id"] = frame_id
        if faces_detected is not None:
            state["faces_detected"] = faces_detected
        if pipeline_state:
            state["state"] = pipeline_state
        entries.append(state)
    
    # Extra entries
    if extra:
        entries.append({
            "type": "extra",
            "timestamp": timestamp,
            **extra
        })
    
    return entries
