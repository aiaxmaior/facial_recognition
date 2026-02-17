"""
Socket.IO client for real-time enrollment push from the IoT broker.

Maintains a persistent Socket.IO connection and receives enrollment.publish
messages containing face embeddings. Uses the same [header, data] message
format as all other broker communication.

Usage:
    from iot_integration.ws_client import WebSocketClient, WebSocketConfig

    config = WebSocketConfig(
        device_id="jetson-001",
        socketio_url="https://acetaxi-bridge.qryde.net",
        socketio_path="/iot-broker/socket.io",
    )
    client = WebSocketClient(config)
    client.on_enrollment_update = my_handler
    client.start()
    # ...
    client.stop()
"""

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Dict, Any

import socketio

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConfig:
    """Configuration for the Socket.IO enrollment client."""
    device_id: str
    socketio_url: str
    socketio_path: str = "/iot-broker/socket.io"
    api_key: Optional[str] = None


class WebSocketClient:
    """
    Socket.IO client for receiving enrollment pushes from the IoT broker.

    All messages use a single 'message' event with payload = [header, data].
    Routing is by header["command_id"]:
        - "device.register"    (edge -> broker)  sent on connect
        - "response.success"   (broker -> edge)  registration confirmed
        - "enrollment.publish" (broker -> edge)  new face embedding

    Lifecycle:
        1. start()  -> connects on a background thread
        2. On connect -> emits device.register
        3. Listens for enrollment.publish, dispatches to callback
        4. python-socketio handles reconnection automatically
        5. stop()   -> disconnects cleanly
    """

    def __init__(self, config: WebSocketConfig):
        self.config = config

        # Callback for enrollment pushes — receives the data dict
        self.on_enrollment_update: Optional[Callable[[dict], None]] = None

        # Socket.IO client with auto-reconnect
        self._sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,  # unlimited
            reconnection_delay=1,
            reconnection_delay_max=60,
            logger=False,
            engineio_logger=False,
        )

        self._thread: Optional[threading.Thread] = None
        self._connected = False

        # Stats
        self.stats: Dict[str, Any] = {
            "connected": False,
            "enrollments_received": 0,
            "last_message_at": None,
            "last_connect_at": None,
        }

        # Register Socket.IO event handlers
        self._sio.on("connect", self._on_connect)
        self._sio.on("disconnect", self._on_disconnect)
        self._sio.on("message", self._on_message)
        self._sio.on("connect_error", self._on_connect_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to the broker on a background daemon thread."""
        if self._sio.connected:
            logger.warning("Socket.IO client already connected")
            return

        self._thread = threading.Thread(
            target=self._connect,
            daemon=True,
            name="SioClient-Worker",
        )
        self._thread.start()
        logger.info(
            f"Socket.IO client starting: {self.config.socketio_url} "
            f"(path={self.config.socketio_path})"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Disconnect from the broker."""
        if self._sio.connected:
            try:
                self._sio.disconnect()
            except Exception:
                pass

        if self._thread:
            self._thread.join(timeout=timeout)

        self._connected = False
        self.stats["connected"] = False
        logger.info(
            f"Socket.IO client stopped. "
            f"Enrollments received: {self.stats['enrollments_received']}"
        )

    def is_connected(self) -> bool:
        return self._connected

    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats, "connected": self._connected}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Connect to the Socket.IO server (blocking)."""
        try:
            self._sio.connect(
                self.config.socketio_url,
                socketio_path=self.config.socketio_path,
                wait=True,
                wait_timeout=30,
            )
            # Block this thread until disconnected (keeps reconnect alive)
            self._sio.wait()
        except Exception as e:
            logger.error(f"Socket.IO connection failed: {e}")

    # ------------------------------------------------------------------
    # Socket.IO event handlers
    # ------------------------------------------------------------------

    def _on_connect(self) -> None:
        self._connected = True
        self.stats["connected"] = True
        self.stats["last_connect_at"] = datetime.utcnow().isoformat() + "Z"
        logger.info("Socket.IO connected to broker")
        self._send_register()

    def _on_disconnect(self) -> None:
        self._connected = False
        self.stats["connected"] = False
        logger.warning("Socket.IO disconnected from broker")

    def _on_connect_error(self, data) -> None:
        logger.error(f"Socket.IO connect error: {data}")

    def _on_message(self, *args) -> None:
        """
        Handle incoming 'message' event.

        Broker payload is [{"header": {...}}, {"data": {...}}].
        Socket.IO may deliver as two positional args or a single list.
        """
        try:
            # Socket.IO may deliver as two args or one list
            if len(args) == 2:
                raw_header, raw_data = args[0], args[1]
            elif len(args) == 1 and isinstance(args[0], (list, tuple)):
                raw_header, raw_data = args[0][0], args[0][1]
            else:
                logger.warning(f"Unexpected message format: {len(args)} args")
                return

            # Parse if strings
            if isinstance(raw_header, str):
                raw_header = json.loads(raw_header)
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)

            # Unwrap {"header": {...}} and {"data": {...}} wrappers
            header = raw_header.get("header", raw_header) if isinstance(raw_header, dict) else raw_header
            data = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.error(f"Failed to parse broker message: {e}")
            return

        command_id = header.get("command_id", "")
        self.stats["last_message_at"] = datetime.utcnow().isoformat() + "Z"

        logger.debug(f"SIO message: command_id={command_id}")

        if command_id == "response.success":
            logger.info("Device registration confirmed by broker")

        elif command_id == "enrollment.publish":
            self._handle_enrollment(header, data)

        else:
            logger.info(f"Unhandled command_id: {command_id}")

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _send_register(self) -> None:
        """Emit device.register message to the broker."""
        header = {
            "to": "gateway",
            "from": self.config.device_id,
            "source_type": "device",
            "auth_token": self.config.api_key,
            "command_id": "device.register",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        data = {
            "request_id": str(uuid.uuid4()),
            "device_category": "camera",
            "capability": "face_recognition",
        }

        try:
            self._sio.emit("message", [{"header": header}, {"data": data}])
            logger.info(
                f"Sent device.register: device={self.config.device_id}"
            )
        except Exception as e:
            logger.error(f"Failed to send register: {e}")

    def _handle_enrollment(self, header: dict, data: dict) -> None:
        """Process an enrollment.publish message."""
        employee_id = data.get("employee_id")
        embedded_file = data.get("embedded_file")

        if not employee_id or not embedded_file:
            logger.error(
                f"enrollment.publish missing required fields: "
                f"employee_id={bool(employee_id)}, embedded_file={bool(embedded_file)}"
            )
            return

        logger.info(f"Enrollment received: employee_id={employee_id}")
        self.stats["enrollments_received"] += 1

        if self.on_enrollment_update:
            try:
                self.on_enrollment_update(data)
            except Exception as e:
                logger.error(f"Enrollment callback failed: {e}")
