#!/bin/bash
#
# Activate QRaie development environment
#
# Usage:
#   source activate.sh           # Activate venv
#   source activate.sh edge      # Activate and cd to edge-device
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for venv locations (local first, then installed)
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PATH="$SCRIPT_DIR/venv"
elif [ -d "$SCRIPT_DIR/modules/edge-device/venv" ]; then
    VENV_PATH="$SCRIPT_DIR/modules/edge-device/venv"
elif [ -d "/opt/qraie/facial_recognition/venv" ]; then
    VENV_PATH="/opt/qraie/facial_recognition/venv"
else
    echo "Error: No venv found. Checked:"
    echo "  - $SCRIPT_DIR/venv"
    echo "  - $SCRIPT_DIR/modules/edge-device/venv"
    echo "  - /opt/qraie/facial_recognition/venv"
    echo ""
    echo "Create a venv:"
    echo "  python3 -m venv venv && source venv/bin/activate"
    return 1 2>/dev/null || exit 1
fi

# Activate
echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Set config path for convenience (local first, then installed)
if [ -f "$SCRIPT_DIR/modules/edge-device/config/config.json" ]; then
    export CONFIG_PATH="$SCRIPT_DIR/modules/edge-device/config/config.json"
elif [ -f "/opt/qraie/config/device_config.json" ]; then
    export CONFIG_PATH="/opt/qraie/config/device_config.json"
else
    export CONFIG_PATH="$SCRIPT_DIR/modules/edge-device/config/config.json"
fi

# Navigate to module if specified
case "$1" in
    edge|edge-device)
        cd "$SCRIPT_DIR/modules/edge-device"
        echo "Changed to: $(pwd)"
        ;;
    iot|iot-integration)
        cd "$SCRIPT_DIR/modules/edge-device/iot_integration"
        echo "Changed to: $(pwd)"
        ;;
esac

echo ""
echo "Environment ready. CONFIG_PATH=$CONFIG_PATH"
echo ""
echo "Quick commands:"
echo "  cd modules/edge-device && python src/device_ctl.py status"
echo "  cd modules/edge-device && python src/device_ctl.py register"
echo "  cd modules/edge-device && python src/device_ctl.py heartbeat"
