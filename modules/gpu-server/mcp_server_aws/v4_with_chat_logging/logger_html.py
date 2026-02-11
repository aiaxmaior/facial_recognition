html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Client Logs - Real-time Monitor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 25px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0 0 15px 0;
            color: #2d3748;
            font-size: 28px;
            font-weight: 600;
        }
        .subtitle {
            color: #718096;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        button:hover {
            background: #5a67d8;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        button.danger {
            background: #f56565;
        }
        button.danger:hover {
            background: #e53e3e;
        }
        button.success {
            background: #48bb78;
        }
        button.success:hover {
            background: #38a169;
        }
        .status {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .status::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .status.connected {
            background: #c6f6d5;
            color: #22543d;
        }
        .status.connected::before {
            background: #48bb78;
            animation: pulse 2s infinite;
        }
        .status.disconnected {
            background: #fed7d7;
            color: #742a2a;
        }
        .status.disconnected::before {
            background: #f56565;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .stats {
            margin-top: 15px;
            display: flex;
            gap: 20px;
            font-size: 14px;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        .stat-label {
            color: #718096;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            color: #2d3748;
            font-size: 20px;
            font-weight: 700;
            margin-top: 2px;
        }
        #log-container {
            background: white;
            padding: 20px;
            border-radius: 12px;
            height: calc(100vh - 300px);
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        #log-container::-webkit-scrollbar {
            width: 10px;
        }
        #log-container::-webkit-scrollbar-track {
            background: #f7fafc;
            border-radius: 10px;
        }
        #log-container::-webkit-scrollbar-thumb {
            background: #cbd5e0;
            border-radius: 10px;
        }
        #log-container::-webkit-scrollbar-thumb:hover {
            background: #a0aec0;
        }
        .log-entry {
            display: grid;
            grid-template-columns: 180px 80px 1fr;
            gap: 15px;
            padding: 12px 15px;
            margin-bottom: 8px;
            border-radius: 8px;
            transition: background 0.2s;
            border-left: 3px solid transparent;
            font-size: 14px;
        }
        .log-entry:hover {
            background: #f7fafc;
        }
        .log-entry.new-entry {
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .log-timestamp {
            color: #718096;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            font-weight: 500;
        }
        .log-level {
            font-weight: 700;
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 4px;
            text-align: center;
            width: fit-content;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .log-level.INFO {
            background: #bee3f8;
            color: #2c5282;
        }
        .log-level.DEBUG {
            background: #e9d8fd;
            color: #553c9a;
        }
        .log-level.WARNING {
            background: #feebc8;
            color: #7c2d12;
        }
        .log-level.ERROR {
            background: #fed7d7;
            color: #742a2a;
        }
        .log-level.CRITICAL {
            background: #fc8181;
            color: white;
        }
        .log-message {
            color: #2d3748;
            line-height: 1.6;
            word-break: break-word;
        }
        .log-entry.ERROR {
            border-left-color: #f56565;
            background: #fff5f5;
        }
        .log-entry.CRITICAL {
            border-left-color: #c53030;
            background: #fff5f5;
        }
        .log-entry.WARNING {
            border-left-color: #ed8936;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #a0aec0;
        }
        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        .filter-section {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .filter-btn {
            padding: 6px 12px;
            font-size: 12px;
            background: #edf2f7;
            color: #4a5568;
        }
        .filter-btn.active {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 MCP Client Logs</h1>
            <p class="subtitle">Real-time log monitoring and analysis</p>
            
            <div class="controls">
                <button onclick="clearLogs()" class="danger">Clear Display</button>
                <button onclick="downloadLogs()" class="success">Download Logs</button>
                <button id="auto-scroll-toggle" class="success" onclick="toggleAutoScroll()">Auto-Scroll: ON</button>
                <span class="status disconnected" id="status">Disconnected</span>
            </div>

            <div class="filter-section">
                <span style="color: #718096; font-size: 13px; font-weight: 600;">Filter:</span>
                <button class="filter-btn active" onclick="filterLogs('ALL')">All</button>
                <button class="filter-btn" onclick="filterLogs('INFO')">Info</button>
                <button class="filter-btn" onclick="filterLogs('DEBUG')">Debug</button>
                <button class="filter-btn" onclick="filterLogs('WARNING')">Warning</button>
                <button class="filter-btn" onclick="filterLogs('ERROR')">Error</button>
                <button class="filter-btn" onclick="filterLogs('CRITICAL')">Critical</button>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-label">Total Logs</span>
                    <span class="stat-value" id="log-count">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Errors</span>
                    <span class="stat-value" id="error-count" style="color: #f56565;">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Warnings</span>
                    <span class="stat-value" id="warning-count" style="color: #ed8936;">0</span>
                </div>
            </div>
        </div>
        
        <div id="log-container">
            <div class="empty-state">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
                <h3>No logs yet</h3>
                <p>Waiting for application logs...</p>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let autoScroll = true;
        let logs = [];
        let currentFilter = 'ALL';
        const logContainer = document.getElementById('log-container');
        const statusElement = document.getElementById('status');
        const logCountElement = document.getElementById('log-count');
        const errorCountElement = document.getElementById('error-count');
        const warningCountElement = document.getElementById('warning-count');

        function getBasePath(){
            const currentPath = window.location.pathname;
            return currentPath.replace(/\/logs\/view$/, '');
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const basePath = getBasePath();
            const wsUrl = `${protocol}//${window.location.host}${basePath}/ws/logs`;
            
            console.log('Connecting to WebSocket:', wsUrl);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('WebSocket connected');
                statusElement.textContent = 'Connected';
                statusElement.className = 'status connected';
                loadExistingLogs();
            };

            ws.onmessage = (event) => {
                const logEntry = JSON.parse(event.data);
                addLogEntry(logEntry, true);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                statusElement.textContent = 'Disconnected';
                statusElement.className = 'status disconnected';
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        async function loadExistingLogs() {
            try {
                const basePath = getBasePath();
                const response = await fetch(`${basePath}/logs?limit=200`);
                const data = await response.json();
                if (data.logs && data.logs.length > 0) {
                    clearEmptyState();
                    data.logs.forEach(log => addLogEntry(log, false, false));
                }
            } catch (error) {
                console.error('Failed to load existing logs:', error);
            }
        }

        function clearEmptyState() {
            const emptyState = logContainer.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }
        }

        function addLogEntry(logEntry, shouldScroll = true, animate = true) {
            clearEmptyState();
            logs.push(logEntry);
            
            const level = logEntry.level || 'INFO';
            
            // Update stats
            logCountElement.textContent = logs.length;
            const errorCount = logs.filter(l => l.level === 'ERROR' || l.level === 'CRITICAL').length;
            const warningCount = logs.filter(l => l.level === 'WARNING').length;
            errorCountElement.textContent = errorCount;
            warningCountElement.textContent = warningCount;
            
            // Check filter
            if (currentFilter !== 'ALL' && level !== currentFilter) {
                return;
            }
            
            const div = document.createElement('div');
            div.className = `log-entry ${level}`;
            if (animate) {
                div.classList.add('new-entry');
            }
            div.dataset.level = level;
            
            const timestamp = logEntry.timestamp_str || new Date(logEntry.timestamp * 1000).toLocaleString();
            const message = escapeHtml(logEntry.message);
            
            div.innerHTML = `
                <div class="log-timestamp">${timestamp}</div>
                <div class="log-level ${level}">${level}</div>
                <div class="log-message">${message}</div>
            `;
            
            logContainer.appendChild(div);
            
            if (autoScroll && shouldScroll) {
                logContainer.scrollTop = logContainer.scrollHeight;
            }
        }

        function clearLogs() {
            logs = [];
            logContainer.innerHTML = '<div class="empty-state"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg><h3>No logs to display</h3><p>All logs have been cleared</p></div>';
            logCountElement.textContent = '0';
            errorCountElement.textContent = '0';
            warningCountElement.textContent = '0';
        }

        function downloadLogs() {
            const logText = logs.map(log => {
                const timestamp = log.timestamp_str || new Date(log.timestamp * 1000).toISOString();
                const level = log.level || 'INFO';
                return `${timestamp} | ${level.padEnd(8)} | ${log.message}`;
            }).join('\\n');
            
            const blob = new Blob([logText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mcp-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }

        function toggleAutoScroll() {
            autoScroll = !autoScroll;
            const button = document.getElementById('auto-scroll-toggle');
            if (autoScroll) {
                button.textContent = 'Auto-Scroll: ON';
                button.classList.add('success');
                logContainer.scrollTop = logContainer.scrollHeight;
            } else {
                button.textContent = 'Auto-Scroll: OFF';
                button.classList.remove('success');
            }
        }

        function filterLogs(level) {
            currentFilter = level;
            
            // Update active button
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Filter display
            const entries = logContainer.querySelectorAll('.log-entry');
            entries.forEach(entry => {
                if (level === 'ALL' || entry.dataset.level === level) {
                    entry.style.display = 'grid';
                } else {
                    entry.style.display = 'none';
                }
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Connect on page load
        connectWebSocket();
    </script>
</body>
</html>
"""
