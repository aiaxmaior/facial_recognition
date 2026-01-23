#!/bin/bash
#
# Install QRaie Facial Recognition Service
#
# Usage:
#   sudo ./install_service.sh
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="/opt/qraie"
SERVICE_USER="qraie"

# Check root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Run as root (sudo)${NC}"
    exit 1
fi

echo -e "${GREEN}Installing QRaie Facial Recognition Service${NC}"

# Create user if not exists
if ! id "$SERVICE_USER" &>/dev/null; then
    echo -e "${YELLOW}Creating service user: $SERVICE_USER${NC}"
    useradd -r -m -s /bin/bash "$SERVICE_USER"
    usermod -aG video "$SERVICE_USER"
fi

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$INSTALL_DIR/facial_recognition"
mkdir -p "$INSTALL_DIR/data/enrollments"
mkdir -p "$INSTALL_DIR/data/video_buffer"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/config"

# Copy application files
echo -e "${YELLOW}Copying application files...${NC}"
cp "$PROJECT_DIR/main.py" "$INSTALL_DIR/facial_recognition/"
cp "$PROJECT_DIR/video_buffer.py" "$INSTALL_DIR/facial_recognition/"

# Copy iot_integration module
if [ -d "$(dirname "$PROJECT_DIR")/iot_integration" ]; then
    cp -r "$(dirname "$PROJECT_DIR")/iot_integration" "$INSTALL_DIR/facial_recognition/"
fi

# Copy config if not exists
if [ ! -f "$INSTALL_DIR/config/device_config.json" ]; then
    if [ -f "$PROJECT_DIR/config/device_config.json" ]; then
        cp "$PROJECT_DIR/config/device_config.json" "$INSTALL_DIR/config/"
    else
        echo -e "${YELLOW}Creating default config...${NC}"
        cat > "$INSTALL_DIR/config/device_config.json" << 'EOF'
{
    "device_id": "jetson-001",
    "broker_url": "https://acetaxi-bridge.qryde.net/iot-broker/api",
    "camera": {
        "rtsp_url": "rtsp://admin:PASSWORD@10.42.0.159/Preview_01_sub",
        "fps": 25
    },
    "recognition": {
        "model": "ArcFace",
        "detector_backend": "yolov8",
        "distance_threshold": 0.35
    },
    "validation": {
        "confirmation_frames": 5,
        "consistency_threshold": 0.8,
        "cooldown_seconds": 30
    },
    "video_buffer": {
        "enabled": true,
        "duration_seconds": 15,
        "buffer_path": "/opt/qraie/data/video_buffer"
    },
    "sync": {
        "enrollment_db_path": "/opt/qraie/data/enrollments/enrollments.db"
    },
    "heartbeat": {
        "interval_seconds": 30
    }
}
EOF
    fi
fi

# Setup Python venv if not exists
if [ ! -d "$INSTALL_DIR/facial_recognition/venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv "$INSTALL_DIR/facial_recognition/venv"
    
    source "$INSTALL_DIR/facial_recognition/venv/bin/activate"
    pip install --upgrade pip
    pip install wheel
    pip install requests websocket-client numpy opencv-python-headless pillow pydantic deepface
    deactivate
fi

# Set ownership
echo -e "${YELLOW}Setting permissions...${NC}"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# Install systemd service
echo -e "${YELLOW}Installing systemd service...${NC}"
cp "$PROJECT_DIR/systemd/qraie-facial.service" /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Service installed at: /etc/systemd/system/qraie-facial.service"
echo "Application dir: $INSTALL_DIR"
echo "Config file: $INSTALL_DIR/config/device_config.json"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit config: sudo nano $INSTALL_DIR/config/device_config.json"
echo "   - Set your camera RTSP URL and password"
echo "   - Set your device_id"
echo ""
echo "2. Enable and start the service:"
echo "   sudo systemctl enable qraie-facial"
echo "   sudo systemctl start qraie-facial"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status qraie-facial"
echo "   sudo journalctl -u qraie-facial -f"
echo ""
