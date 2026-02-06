#!/bin/bash
#
# Install QRaie Facial Recognition Service + Health Monitoring
#
# Installs:
#   - qraie-facial.service   (main pipeline)
#   - qraie-health.service   (health check daemon)
#   - qraie-health-check.*   (periodic health check timer)
#
# Also ensures SSH is enabled and the device boots into an operable state.
#
# Usage:
#   sudo ./install_service.sh
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  QRaie Edge Device - Full Installation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

#----------------------------------------------------------------------
# 1. Create service user
#----------------------------------------------------------------------
echo -e "${BLUE}[1/8] Service User${NC}"

if ! id "$SERVICE_USER" &>/dev/null; then
    echo -e "${YELLOW}Creating service user: $SERVICE_USER${NC}"
    useradd -r -m -s /bin/bash "$SERVICE_USER"
    usermod -aG video "$SERVICE_USER"
    echo -e "${GREEN}Created user $SERVICE_USER${NC}"
else
    echo -e "${GREEN}User $SERVICE_USER already exists${NC}"
fi

#----------------------------------------------------------------------
# 2. Create directories
#----------------------------------------------------------------------
echo -e "${BLUE}[2/8] Directories${NC}"

mkdir -p "$INSTALL_DIR/facial_recognition/src"
mkdir -p "$INSTALL_DIR/facial_recognition/iot_integration/schemas"
mkdir -p "$INSTALL_DIR/facial_recognition/models"
mkdir -p "$INSTALL_DIR/facial_recognition/logs"
mkdir -p "$INSTALL_DIR/data/enrollments"
mkdir -p "$INSTALL_DIR/data/video_buffer"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/config"

echo -e "${GREEN}Directories created under $INSTALL_DIR${NC}"

#----------------------------------------------------------------------
# 3. Copy application files
#----------------------------------------------------------------------
echo -e "${BLUE}[3/8] Application Files${NC}"

# Core source files
for f in main.py video_buffer.py device_ctl.py face_recognition.py \
         trt_face_detector.py ds_pipeline.py health_check.py; do
    if [ -f "$PROJECT_DIR/src/$f" ]; then
        cp "$PROJECT_DIR/src/$f" "$INSTALL_DIR/facial_recognition/src/"
        echo "  Copied src/$f"
    fi
done

# Legacy main.py in root (for backward compat with old service file)
if [ -f "$PROJECT_DIR/main.py" ]; then
    cp "$PROJECT_DIR/main.py" "$INSTALL_DIR/facial_recognition/"
fi

# IoT integration module
if [ -d "$PROJECT_DIR/iot_integration" ]; then
    echo -e "${YELLOW}Copying iot_integration module...${NC}"
    cp -r "$PROJECT_DIR/iot_integration/" "$INSTALL_DIR/facial_recognition/iot_integration/"
else
    echo -e "${RED}Warning: iot_integration not found at $PROJECT_DIR/iot_integration${NC}"
fi

# Models directory
if [ -d "$PROJECT_DIR/models" ]; then
    cp -r "$PROJECT_DIR/models/" "$INSTALL_DIR/facial_recognition/models/"
    echo "  Copied models/"
fi

# Requirements
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    cp "$PROJECT_DIR/requirements.txt" "$INSTALL_DIR/facial_recognition/"
fi

echo -e "${GREEN}Application files installed${NC}"

#----------------------------------------------------------------------
# 4. Copy config (if not exists - don't overwrite existing)
#----------------------------------------------------------------------
echo -e "${BLUE}[4/8] Configuration${NC}"

if [ ! -f "$INSTALL_DIR/config/device_config.json" ]; then
    if [ -f "$PROJECT_DIR/config/config.json" ]; then
        cp "$PROJECT_DIR/config/config.json" "$INSTALL_DIR/config/device_config.json"
        echo -e "${GREEN}Config copied from project${NC}"
    else
        echo -e "${YELLOW}Creating default config...${NC}"
        DEVICE_ID=$(hostname | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
        cat > "$INSTALL_DIR/config/device_config.json" << EOF
{
    "device_id": "$DEVICE_ID",
    "broker_url": "https://acetaxi-bridge.qryde.net/iot-broker/api",
    "camera": {
        "rtsp_url": "rtsp://admin:PASSWORD@CAMERA_IP/Preview_01_sub",
        "fps": 25
    },
    "recognition": {
        "model": "ArcFace",
        "detector_backend": "yolov8n",
        "distance_threshold": 0.55
    },
    "validation": {
        "confirmation_frames": 3,
        "consistency_threshold": 0.6,
        "cooldown_seconds": 2
    },
    "video_buffer": {
        "enabled": false,
        "duration_seconds": 15,
        "buffer_path": "/opt/qraie/data/video_buffer"
    },
    "sync": {
        "enrollment_db_path": "/opt/qraie/data/enrollments/enrollments.db"
    },
    "heartbeat": {
        "interval_seconds": 30
    },
    "health_check": {
        "auto_remediate": true,
        "thresholds": {
            "cpu_warn_percent": 85,
            "cpu_crit_percent": 95,
            "memory_warn_percent": 85,
            "memory_crit_percent": 95,
            "disk_warn_percent": 85,
            "disk_crit_percent": 95,
            "temp_warn_c": 75,
            "temp_crit_c": 85
        }
    }
}
EOF
        echo -e "${GREEN}Default config created${NC}"
    fi
else
    echo -e "${GREEN}Config already exists (not overwritten)${NC}"
fi

#----------------------------------------------------------------------
# 5. Python virtual environment
#----------------------------------------------------------------------
echo -e "${BLUE}[5/8] Python Virtual Environment${NC}"

VENV_DIR="$INSTALL_DIR/facial_recognition/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"

    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install wheel -q

    if [ -f "$INSTALL_DIR/facial_recognition/requirements.txt" ]; then
        pip install -r "$INSTALL_DIR/facial_recognition/requirements.txt" -q
    else
        pip install requests websocket-client numpy opencv-python-headless pillow pydantic deepface -q
    fi

    deactivate
    echo -e "${GREEN}Python venv created${NC}"
else
    echo -e "${GREEN}Python venv already exists${NC}"
fi

#----------------------------------------------------------------------
# 6. Set permissions
#----------------------------------------------------------------------
echo -e "${BLUE}[6/8] Permissions${NC}"

chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
# Health check daemon needs root for remediation (service restarts)
# so health service files run as root, but the main service runs as qraie

echo -e "${GREEN}Permissions set${NC}"

#----------------------------------------------------------------------
# 7. Install systemd services
#----------------------------------------------------------------------
echo -e "${BLUE}[7/8] Systemd Services${NC}"

# Main facial recognition service
cp "$PROJECT_DIR/systemd/qraie-facial.service" /etc/systemd/system/
echo "  Installed qraie-facial.service"

# Health check daemon
if [ -f "$PROJECT_DIR/systemd/qraie-health.service" ]; then
    cp "$PROJECT_DIR/systemd/qraie-health.service" /etc/systemd/system/
    echo "  Installed qraie-health.service"
fi

# Periodic health check timer + one-shot service
if [ -f "$PROJECT_DIR/systemd/qraie-health-check.timer" ]; then
    cp "$PROJECT_DIR/systemd/qraie-health-check.timer" /etc/systemd/system/
    cp "$PROJECT_DIR/systemd/qraie-health-check.service" /etc/systemd/system/
    echo "  Installed qraie-health-check.timer"
fi

# Reload systemd
systemctl daemon-reload

# Enable all services for boot
systemctl enable qraie-facial
echo "  Enabled qraie-facial at boot"

if [ -f "/etc/systemd/system/qraie-health.service" ]; then
    systemctl enable qraie-health
    echo "  Enabled qraie-health at boot"
fi

if [ -f "/etc/systemd/system/qraie-health-check.timer" ]; then
    systemctl enable qraie-health-check.timer
    echo "  Enabled health check timer at boot"
fi

echo -e "${GREEN}Systemd services installed and enabled${NC}"

#----------------------------------------------------------------------
# 8. Boot hardening (SSH, network, etc.)
#----------------------------------------------------------------------
echo -e "${BLUE}[8/8] Boot Hardening${NC}"

# Ensure SSH is enabled
for svc in ssh sshd; do
    if systemctl list-unit-files "$svc.service" &>/dev/null 2>&1; then
        systemctl enable "$svc" 2>/dev/null || true
        systemctl start "$svc" 2>/dev/null || true
        echo "  SSH ($svc) enabled and started"
        break
    fi
done

# Set boot target to CLI
systemctl set-default multi-user.target 2>/dev/null || true
echo "  Boot target set to multi-user (CLI)"

echo -e "${GREEN}Boot hardening complete${NC}"

#----------------------------------------------------------------------
# Summary
#----------------------------------------------------------------------

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Install Dir:   $INSTALL_DIR"
echo "  Config:        $INSTALL_DIR/config/device_config.json"
echo "  Python venv:   $VENV_DIR"
echo ""
echo "  Services installed:"
echo "    qraie-facial          - Main facial recognition pipeline"
echo "    qraie-health          - Health check daemon (self-monitoring)"
echo "    qraie-health-check    - Periodic health check (every 5 min)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1. Edit config (set camera URL, etc.):"
echo "   sudo nano $INSTALL_DIR/config/device_config.json"
echo ""
echo "2. Configure camera (auto-discovery):"
echo "   cd $INSTALL_DIR/facial_recognition"
echo "   source venv/bin/activate"
echo "   python src/device_ctl.py camera"
echo ""
echo "3. Start services:"
echo "   sudo systemctl start qraie-facial"
echo "   sudo systemctl start qraie-health"
echo "   sudo systemctl start qraie-health-check.timer"
echo ""
echo "4. Verify boot readiness:"
echo "   sudo $SCRIPT_DIR/setup_boot.sh --check"
echo ""
echo "5. Monitor:"
echo "   sudo journalctl -u qraie-facial -f"
echo "   sudo journalctl -u qraie-health -f"
echo "   python src/health_check.py --config $INSTALL_DIR/config/device_config.json"
echo ""
echo "6. Reboot to verify everything starts:"
echo "   sudo reboot"
echo ""
