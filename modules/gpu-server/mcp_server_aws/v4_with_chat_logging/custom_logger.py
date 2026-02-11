import time
from collections import deque
from typing import List


class LogHandler:
    """Custom handler to capture logs in memory and broadcast to websockets"""

    def __init__(self, log_queue: deque, websocket_connections: List):
        self.websocket_connections = websocket_connections
        self.log_storage = log_queue

    def write(self, message):
        record = message.strip()
        if record:
            # Parse the log message to extract components
            parts = record.split(" | ", 2)
            if len(parts) >= 3:
                timestamp_str, level, msg = parts
                log_entry = {
                    "timestamp": time.time(),
                    "timestamp_str": timestamp_str,
                    "level": level.strip(),
                    "message": msg.strip(),
                }
            else:
                log_entry = {
                    "timestamp": time.time(),
                    "timestamp_str": "",
                    "level": "INFO",
                    "message": record,
                }

            self.log_storage.append(log_entry)
            # Broadcast to all connected websockets
            for ws in self.websocket_connections[:]:  # Create a copy to iterate safely
                try:
                    import asyncio

                    asyncio.create_task(ws.send_json(log_entry))
                except Exception:
                    pass  # Remove disconnected clients silently
