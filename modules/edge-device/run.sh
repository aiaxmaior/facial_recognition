#!/bin/bash
#
# QRaie Edge Device - One-step launcher
#
# Usage:
#   ./run.sh              # Full startup
#   ./run.sh --skip-camera # Skip camera check (testing)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Find venv (local first, then opt)
if [ -d "$REPO_ROOT/venv" ]; then
    VENV_PATH="$REPO_ROOT/venv"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PATH="$SCRIPT_DIR/venv"
elif [ -d "/opt/qraie/facial_recognition/venv" ]; then
    VENV_PATH="/opt/qraie/facial_recognition/venv"
else
    echo "Error: No venv found!"
    echo "Create one: cd $REPO_ROOT && python3 -m venv venv"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Set config path
export CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/config/config.json}"

echo "========================================"
echo "  QRaie Edge Device Launcher"
echo "========================================"
echo "  Venv: $VENV_PATH"
echo "  Config: $CONFIG_PATH"
echo "========================================"
echo ""

# Run the device
cd "$SCRIPT_DIR"
python src/device_ctl.py run "$@"
