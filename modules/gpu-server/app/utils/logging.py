"""
Logging utilities for GPU Server

Includes WebSocket log handler and HTML log viewer.
"""

import asyncio
from typing import List, Deque
from fastapi import WebSocket


class LogHandler:
    """
    Custom log handler that stores logs and broadcasts to WebSocket clients.
    
    Compatible with loguru. Safe to call from both async and sync (thread pool) contexts.
    """
    
    def __init__(self, log_queue: Deque, websocket_connections: List[WebSocket]):
        self.log_queue = log_queue
        self.websocket_connections = websocket_connections
    
    def write(self, message: str):
        """Write log message to queue and broadcast to WebSocket clients"""
        log_entry = message.strip()
        if log_entry:
            self.log_queue.append(log_entry)
            
            # Broadcast to WebSocket clients (non-blocking, thread-safe)
            if self.websocket_connections:
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon_threadsafe(
                        lambda entry=log_entry: asyncio.ensure_future(self._broadcast(entry))
                    )
                except RuntimeError:
                    # No running event loop (called from thread pool) -- skip broadcast
                    # Logs are still stored in the deque and accessible via /logs endpoint
                    pass
    
    async def _broadcast(self, message: str):
        """Broadcast message to all connected WebSocket clients"""
        disconnected = []
        
        for ws in self.websocket_connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            if ws in self.websocket_connections:
                self.websocket_connections.remove(ws)


def get_log_viewer_html() -> str:
    """Return HTML content for log viewer UI"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Server - Log Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background-color: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background-color: #252526;
            padding: 12px 20px;
            border-bottom: 1px solid #3c3c3c;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 16px;
            font-weight: 500;
            color: #cccccc;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #f44336;
        }
        .status-dot.connected {
            background-color: #4caf50;
        }
        .controls {
            display: flex;
            gap: 10px;
        }
        button {
            background-color: #0e639c;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }
        button:hover {
            background-color: #1177bb;
        }
        .log-container {
            flex: 1;
            overflow-y: auto;
            padding: 10px 20px;
        }
        .log-entry {
            padding: 2px 0;
            font-size: 13px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .log-entry.INFO { color: #3794ff; }
        .log-entry.DEBUG { color: #808080; }
        .log-entry.WARNING { color: #cca700; }
        .log-entry.ERROR { color: #f44747; }
        .log-entry.CRITICAL { color: #ff0000; font-weight: bold; }
        .filter-bar {
            background-color: #252526;
            padding: 8px 20px;
            border-bottom: 1px solid #3c3c3c;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .filter-bar input {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            color: #d4d4d4;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            width: 200px;
        }
        .filter-bar label {
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .filter-bar input[type="checkbox"] {
            width: auto;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>GPU Server - Log Viewer</h1>
        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Disconnected</span>
        </div>
        <div class="controls">
            <button onclick="clearLogs()">Clear</button>
            <button onclick="toggleAutoScroll()">Auto-scroll: ON</button>
        </div>
    </div>
    <div class="filter-bar">
        <input type="text" id="filterInput" placeholder="Filter logs..." oninput="filterLogs()">
        <label><input type="checkbox" id="showDebug" checked onchange="filterLogs()"> DEBUG</label>
        <label><input type="checkbox" id="showInfo" checked onchange="filterLogs()"> INFO</label>
        <label><input type="checkbox" id="showWarning" checked onchange="filterLogs()"> WARNING</label>
        <label><input type="checkbox" id="showError" checked onchange="filterLogs()"> ERROR</label>
    </div>
    <div class="log-container" id="logContainer"></div>
    
    <script>
        let autoScroll = true;
        let ws = null;
        const logContainer = document.getElementById('logContainer');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/ws/logs`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                statusDot.classList.add('connected');
                statusText.textContent = 'Connected';
            };
            
            ws.onclose = () => {
                statusDot.classList.remove('connected');
                statusText.textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };
            
            ws.onmessage = (event) => {
                addLogEntry(event.data);
            };
        }
        
        function addLogEntry(message) {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = message;
            
            // Determine log level
            if (message.includes('| DEBUG |')) entry.classList.add('DEBUG');
            else if (message.includes('| INFO |')) entry.classList.add('INFO');
            else if (message.includes('| WARNING |')) entry.classList.add('WARNING');
            else if (message.includes('| ERROR |')) entry.classList.add('ERROR');
            else if (message.includes('| CRITICAL |')) entry.classList.add('CRITICAL');
            
            entry.dataset.level = entry.classList[1] || 'INFO';
            entry.dataset.text = message.toLowerCase();
            
            logContainer.appendChild(entry);
            
            if (autoScroll) {
                logContainer.scrollTop = logContainer.scrollHeight;
            }
            
            filterLogs();
        }
        
        function clearLogs() {
            logContainer.innerHTML = '';
        }
        
        function toggleAutoScroll() {
            autoScroll = !autoScroll;
            event.target.textContent = `Auto-scroll: ${autoScroll ? 'ON' : 'OFF'}`;
        }
        
        function filterLogs() {
            const filter = document.getElementById('filterInput').value.toLowerCase();
            const showDebug = document.getElementById('showDebug').checked;
            const showInfo = document.getElementById('showInfo').checked;
            const showWarning = document.getElementById('showWarning').checked;
            const showError = document.getElementById('showError').checked;
            
            document.querySelectorAll('.log-entry').forEach(entry => {
                const level = entry.dataset.level;
                const text = entry.dataset.text;
                
                let visible = true;
                
                if (level === 'DEBUG' && !showDebug) visible = false;
                if (level === 'INFO' && !showInfo) visible = false;
                if (level === 'WARNING' && !showWarning) visible = false;
                if ((level === 'ERROR' || level === 'CRITICAL') && !showError) visible = false;
                
                if (filter && !text.includes(filter)) visible = false;
                
                entry.style.display = visible ? 'block' : 'none';
            });
        }
        
        // Load existing logs
        fetch('/api/logs?limit=500')
            .then(response => response.json())
            .then(data => {
                data.logs.forEach(log => addLogEntry(log));
            });
        
        connect();
    </script>
</body>
</html>
'''
