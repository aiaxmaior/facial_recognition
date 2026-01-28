#!/bin/bash
#
# Jetson Orin Nano Headless Setup Script
# Configures the Jetson for production edge deployment
#
# Usage:
#   sudo ./setup_jetson_headless.sh [--full-removal] [--skip-gui] [--skip-dev]
#
# Options:
#   --full-removal  Remove GUI packages completely (saves 3-5GB)
#   --skip-gui      Skip GUI removal step
#   --skip-dev      Skip dev package removal step
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_USER="qraie"
INSTALL_DIR="/opt/qraie"
JETPACK_VERSION=""

# Parse arguments
FULL_REMOVAL=false
SKIP_GUI=false
SKIP_DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full-removal)
            FULL_REMOVAL=true
            shift
            ;;
        --skip-gui)
            SKIP_GUI=true
            shift
            ;;
        --skip-dev)
            SKIP_DEV=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Detect JetPack version
detect_jetpack_version() {
    echo -e "${YELLOW}Detecting JetPack version...${NC}"
    
    if [ -f /etc/nv_tegra_release ]; then
        JETPACK_VERSION=$(cat /etc/nv_tegra_release | head -1)
        echo -e "${GREEN}Detected: $JETPACK_VERSION${NC}"
    else
        echo -e "${YELLOW}Warning: Could not detect JetPack version${NC}"
    fi
    
    # Check L4T version
    if command -v dpkg &> /dev/null; then
        L4T_VERSION=$(dpkg -l | grep nvidia-l4t-core | awk '{print $3}' | head -1)
        if [ -n "$L4T_VERSION" ]; then
            echo -e "${GREEN}L4T Version: $L4T_VERSION${NC}"
        fi
    fi
}

# Create service user
create_service_user() {
    echo -e "${YELLOW}Creating service user: $SERVICE_USER${NC}"
    
    if id "$SERVICE_USER" &>/dev/null; then
        echo -e "${GREEN}User $SERVICE_USER already exists${NC}"
    else
        useradd -r -m -s /bin/bash "$SERVICE_USER"
        usermod -aG video "$SERVICE_USER"
        usermod -aG gpio "$SERVICE_USER"
        echo -e "${GREEN}Created user $SERVICE_USER${NC}"
    fi
}

# Create directory structure
create_directories() {
    echo -e "${YELLOW}Creating directory structure...${NC}"
    
    mkdir -p "$INSTALL_DIR/facial_recognition"
    mkdir -p "$INSTALL_DIR/data/enrollments"
    mkdir -p "$INSTALL_DIR/data/video_buffer"
    mkdir -p "$INSTALL_DIR/logs"
    mkdir -p "$INSTALL_DIR/config"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    
    echo -e "${GREEN}Created directories under $INSTALL_DIR${NC}"
}

# Disable GUI (quick method)
disable_gui_quick() {
    echo -e "${YELLOW}Disabling GUI (quick method)...${NC}"
    
    # Set default target to multi-user (CLI)
    systemctl set-default multi-user.target
    
    # Stop display manager if running
    if systemctl is-active --quiet gdm3; then
        systemctl stop gdm3
        systemctl disable gdm3
    fi
    
    if systemctl is-active --quiet lightdm; then
        systemctl stop lightdm
        systemctl disable lightdm
    fi
    
    echo -e "${GREEN}GUI disabled. Will boot to CLI after reboot.${NC}"
    echo -e "${YELLOW}Note: GUI packages still installed. Use --full-removal to remove them.${NC}"
}

# Remove GUI packages completely
remove_gui_packages() {
    echo -e "${YELLOW}Removing GUI packages (this will save 3-5GB)...${NC}"
    
    # Download package list for JetPack
    PACKAGE_LIST_URL="https://raw.githubusercontent.com/NVIDIA-AI-IOT/jetson-min-disk/master/assets/nvubuntu-focal-packages_only-in-desktop.txt"
    PACKAGE_LIST="/tmp/nvubuntu-focal-packages_only-in-desktop.txt"
    
    if command -v wget &> /dev/null; then
        wget -q "$PACKAGE_LIST_URL" -O "$PACKAGE_LIST" 2>/dev/null || true
    elif command -v curl &> /dev/null; then
        curl -sL "$PACKAGE_LIST_URL" -o "$PACKAGE_LIST" 2>/dev/null || true
    fi
    
    if [ -f "$PACKAGE_LIST" ] && [ -s "$PACKAGE_LIST" ]; then
        echo -e "${YELLOW}Removing packages from list...${NC}"
        apt-get update
        apt-get purge -y $(cat "$PACKAGE_LIST") 2>/dev/null || true
        apt-get autoremove -y
    else
        # Fallback: remove common desktop packages
        echo -e "${YELLOW}Using fallback package list...${NC}"
        apt-get purge -y \
            ubuntu-desktop \
            gnome-shell \
            gdm3 \
            nautilus \
            gnome-terminal \
            gedit \
            firefox \
            chromium-browser \
            libreoffice* \
            thunderbird* \
            rhythmbox* \
            totem* \
            2>/dev/null || true
        apt-get autoremove -y
    fi
    
    # CRITICAL: Reinstall network-manager
    echo -e "${YELLOW}Reinstalling network-manager (required for network after reboot)...${NC}"
    apt-get install -y network-manager
    
    # Set default target
    systemctl set-default multi-user.target
    
    echo -e "${GREEN}GUI packages removed.${NC}"
}

# Remove development packages
remove_dev_packages() {
    echo -e "${YELLOW}Removing development packages...${NC}"
    
    # Remove CUDA samples and documentation
    echo "Removing CUDA samples and docs..."
    dpkg -r --force-depends cuda-documentation-* 2>/dev/null || true
    dpkg -r --force-depends cuda-samples-* 2>/dev/null || true
    dpkg -r --force-depends libnvinfer-samples 2>/dev/null || true
    dpkg -r --force-depends libnvinfer-doc 2>/dev/null || true
    dpkg -r --force-depends libvisionworks-samples 2>/dev/null || true
    dpkg -r --force-depends vpi*-samples 2>/dev/null || true
    
    # Remove development packages (headers, etc.)
    echo "Removing -dev packages..."
    DEV_PACKAGES=$(dpkg-query -Wf '${Package}\n' 2>/dev/null | grep -E '\-dev$' || true)
    if [ -n "$DEV_PACKAGES" ]; then
        dpkg -r --force-depends $DEV_PACKAGES 2>/dev/null || true
    fi
    
    # Remove static libraries
    echo "Removing static libraries..."
    find /usr -name 'lib*_static*.a' -delete 2>/dev/null || true
    find /usr -name 'lib*_static.a' -delete 2>/dev/null || true
    
    # Clean up
    apt-get autoremove -y
    apt-get clean
    
    echo -e "${GREEN}Development packages removed.${NC}"
}

# Install runtime dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing runtime dependencies...${NC}"
    
    apt-get update
    apt-get install -y \
        python3-pip \
        python3-venv \
        libopencv-dev \
        curl \
        jq \
        iptables-persistent \
        network-manager
    
    echo -e "${GREEN}Dependencies installed.${NC}"
}

# Configure network (from reolink setup)
configure_network() {
    echo -e "${YELLOW}Configuring network for PoE camera setup...${NC}"
    
    # Enable IP forwarding
    echo "net.ipv4.ip_forward=1" > /etc/sysctl.d/99-ip-forward.conf
    sysctl -p /etc/sysctl.d/99-ip-forward.conf
    
    echo -e "${GREEN}IP forwarding enabled.${NC}"
    echo -e "${YELLOW}Note: Run 'nmcli con add type ethernet con-name InternetShare ifname eno1 ipv4.method shared' to share internet with camera${NC}"
}

# Setup Python virtual environment
setup_python_venv() {
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    
    VENV_DIR="$INSTALL_DIR/facial_recognition/venv"
    
    python3 -m venv "$VENV_DIR"
    
    # Activate and install base packages
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install wheel
    
    # Install required packages
    pip install \
        requests \
        websocket-client \
        numpy \
        opencv-python-headless \
        pillow \
        pydantic
    
    deactivate
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$VENV_DIR"
    
    echo -e "${GREEN}Python venv created at $VENV_DIR${NC}"
}

# Create default config
create_default_config() {
    echo -e "${YELLOW}Creating default configuration...${NC}"
    
    CONFIG_FILE="$INSTALL_DIR/config/device_config.json"
    
    # Get hostname for device_id
    DEVICE_ID=$(hostname | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
    
    cat > "$CONFIG_FILE" << EOF
{
    "device_id": "$DEVICE_ID",
    "broker_url": "https://acetaxi-bridge.qryde.net/iot-broker/api",
    "archive_url": "https://archive.qryde.net/api/archive",
    
    "camera": {
        "rtsp_url": "rtsp://admin:PASSWORD@CAMERA_IP/Preview_01_sub",
        "resolution": [1280, 720],
        "fps": 25
    },
    
    "recognition": {
        "model": "ArcFace",
        "detector_backend": "yolov8",
        "distance_threshold": 0.35,
        "min_face_size": 60
    },
    
    "validation": {
        "confirmation_frames": 5,
        "consistency_threshold": 0.80,
        "cooldown_seconds": 30
    },
    
    "video_buffer": {
        "enabled": true,
        "duration_seconds": 15,
        "pre_event_seconds": 10,
        "post_event_seconds": 5,
        "buffer_path": "/opt/qraie/data/video_buffer",
        "codec": "h265"
    },
    
    "logging": {
        "level": "INFO",
        "batch_size": 100,
        "batch_timeout_seconds": 30,
        "log_path": "/opt/qraie/logs"
    },
    
    "sync": {
        "interval_minutes": 15,
        "enrollment_db_path": "/opt/qraie/data/enrollments/enrollments.db"
    },
    
    "heartbeat": {
        "interval_seconds": 30
    }
}
EOF
    
    chown "$SERVICE_USER:$SERVICE_USER" "$CONFIG_FILE"
    
    echo -e "${GREEN}Created config at $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Note: Edit the config file to set your camera RTSP URL and password${NC}"
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Jetson Headless Setup Complete${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Service User:    ${YELLOW}$SERVICE_USER${NC}"
    echo -e "Install Dir:     ${YELLOW}$INSTALL_DIR${NC}"
    echo -e "Config File:     ${YELLOW}$INSTALL_DIR/config/device_config.json${NC}"
    echo -e "Python venv:     ${YELLOW}$INSTALL_DIR/facial_recognition/venv${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Edit config: sudo nano $INSTALL_DIR/config/device_config.json"
    echo "2. Copy facial recognition scripts to $INSTALL_DIR/facial_recognition/"
    echo "3. Install the systemd service"
    echo "4. Reboot to apply headless mode"
    echo ""
    if [ "$FULL_REMOVAL" = true ] || [ "$SKIP_GUI" = false ]; then
        echo -e "${RED}IMPORTANT: Reboot required to apply changes${NC}"
        echo "Run: sudo reboot"
    fi
}

# Main execution
main() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Jetson Orin Nano Headless Setup${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    detect_jetpack_version
    create_service_user
    create_directories
    
    # GUI handling
    if [ "$SKIP_GUI" = false ]; then
        if [ "$FULL_REMOVAL" = true ]; then
            remove_gui_packages
        else
            disable_gui_quick
        fi
    fi
    
    # Dev packages
    if [ "$SKIP_DEV" = false ]; then
        remove_dev_packages
    fi
    
    install_dependencies
    configure_network
    setup_python_venv
    create_default_config
    
    print_summary
}

# Run main
main
