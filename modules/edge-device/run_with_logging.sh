#!/bin/bash
# Auto-restart wrapper: tees stdout/stderr to a timestamped log and restarts on exit.
# The pipeline also writes to logs/ by default (see startup message "Log file: ...").
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "=== Pipeline started at $(date) ===" | tee -a "$LOG_FILE"
echo "Tee log: $LOG_FILE (app also writes to $LOG_DIR/events_*.jsonl or device_*.log)"

cd "$SCRIPT_DIR"

while true; do
    echo "=== Starting pipeline at $(date) ===" >> "$LOG_FILE"
    
    # Run pipeline, log output
    python src/device_ctl.py --config config/config.json run 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?
    echo "=== Pipeline exited with code $EXIT_CODE at $(date) ===" >> "$LOG_FILE"
    
    echo "Restarting in 5 seconds..." >> "$LOG_FILE"
    sleep 5
done
