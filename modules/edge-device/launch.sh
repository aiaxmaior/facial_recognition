#!/bin/bash
#===============================================================================
# QRaie Edge Device - Production Launch Script
#===============================================================================
#
# Comprehensive launch script that handles:
#   1. Python virtual environment setup
#   2. Dependencies installation
#   3. Camera auto-discovery (if needed)
#   4. IoT broker registration
#   5. Main pipeline execution
#
# Usage:
#   ./launch.sh                    # Full startup with all checks
#   ./launch.sh --quick            # Skip setup, just run
#   ./launch.sh --discover-camera  # Force camera discovery
#   ./launch.sh --offline          # Skip broker registration
#   ./launch.sh --help             # Show help
#
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default paths
VENV_PATHS=(
    "$REPO_ROOT/venv"
    "$SCRIPT_DIR/venv"
    "/opt/qraie/facial_recognition/venv"
)
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/config/config.json}"
LOG_DIR="$SCRIPT_DIR/logs"
STATUS_FILE="$SCRIPT_DIR/.device_status"

# Flags
QUICK_MODE=false
DISCOVER_CAMERA=false
OFFLINE_MODE=false
DEBUG_MODE=false

#===============================================================================
# Helper Functions
#===============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

save_status() {
    local key="$1"
    local value="$2"
    local timestamp=$(date -Iseconds)
    
    # Create status file if needed
    if [ ! -f "$STATUS_FILE" ]; then
        echo "{}" > "$STATUS_FILE"
    fi
    
    # Update status (using Python for JSON handling)
    python3 -c "
import json
with open('$STATUS_FILE', 'r') as f:
    status = json.load(f)
status['$key'] = '$value'
status['${key}_timestamp'] = '$timestamp'
with open('$STATUS_FILE', 'w') as f:
    json.dump(status, f, indent=2)
" 2>/dev/null || echo "{\"$key\": \"$value\"}" > "$STATUS_FILE"
}

get_status() {
    local key="$1"
    if [ -f "$STATUS_FILE" ]; then
        python3 -c "
import json
with open('$STATUS_FILE', 'r') as f:
    status = json.load(f)
print(status.get('$key', 'unknown'))
" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

show_help() {
    echo "QRaie Edge Device - Production Launch Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick, -q         Skip setup checks, just run pipeline"
    echo "  --discover-camera   Force camera auto-discovery"
    echo "  --offline           Skip IoT broker registration"
    echo "  --debug             Enable debug logging"
    echo "  --config PATH       Path to config file"
    echo "  --help, -h          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                          # Full startup"
    echo "  $0 --quick                  # Quick start (skip checks)"
    echo "  $0 --discover-camera        # Re-discover camera"
    echo "  $0 --offline --debug        # Local testing with debug"
    echo ""
}

#===============================================================================
# Parse Arguments
#===============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --discover-camera)
            DISCOVER_CAMERA=true
            shift
            ;;
        --offline)
            OFFLINE_MODE=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

#===============================================================================
# Main Setup
#===============================================================================

log_header "QRaie Edge Device - Startup"
echo "  Time: $(date)"
echo "  Mode: $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full")"
echo "  Config: $CONFIG_PATH"
echo "============================================================"

#-------------------------------------------------------------------------------
# Step 1: Find or Create Virtual Environment
#-------------------------------------------------------------------------------

if [ "$QUICK_MODE" = false ]; then
    log_header "Step 1/6: Virtual Environment"
    
    VENV_PATH=""
    for vp in "${VENV_PATHS[@]}"; do
        if [ -d "$vp" ] && [ -f "$vp/bin/activate" ]; then
            VENV_PATH="$vp"
            break
        fi
    done
    
    if [ -z "$VENV_PATH" ]; then
        log_warn "No existing venv found. Creating one..."
        VENV_PATH="$SCRIPT_DIR/venv"
        
        log_info "Creating virtual environment at $VENV_PATH"
        python3 -m venv "$VENV_PATH"
        
        if [ ! -f "$VENV_PATH/bin/activate" ]; then
            log_error "Failed to create virtual environment!"
            exit 1
        fi
        
        log_success "Virtual environment created"
        
        # Mark that we need to install deps
        INSTALL_DEPS=true
    else
        log_success "Found venv: $VENV_PATH"
        INSTALL_DEPS=false
    fi
else
    # Quick mode - find existing venv
    for vp in "${VENV_PATHS[@]}"; do
        if [ -d "$vp" ] && [ -f "$vp/bin/activate" ]; then
            VENV_PATH="$vp"
            break
        fi
    done
    
    if [ -z "$VENV_PATH" ]; then
        log_error "No venv found! Run without --quick first."
        exit 1
    fi
fi

# Activate venv
source "$VENV_PATH/bin/activate"
log_info "Activated venv: $VENV_PATH"

#-------------------------------------------------------------------------------
# Step 2: Install Dependencies
#-------------------------------------------------------------------------------

if [ "$QUICK_MODE" = false ]; then
    log_header "Step 2/6: Dependencies"
    
    REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
    
    if [ "${INSTALL_DEPS:-false}" = true ] || [ ! -f "$VENV_PATH/.deps_installed" ]; then
        if [ -f "$REQUIREMENTS_FILE" ]; then
            log_info "Installing Python dependencies..."
            pip install --upgrade pip -q
            pip install -r "$REQUIREMENTS_FILE" -q
            
            # Mark dependencies as installed
            touch "$VENV_PATH/.deps_installed"
            log_success "Dependencies installed"
        else
            log_warn "No requirements.txt found at $REQUIREMENTS_FILE"
        fi
    else
        log_success "Dependencies already installed"
    fi
else
    log_info "Skipping dependency check (quick mode)"
fi

#-------------------------------------------------------------------------------
# Step 3: Verify Configuration
#-------------------------------------------------------------------------------

if [ "$QUICK_MODE" = false ]; then
    log_header "Step 3/6: Configuration"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    log_error "Config file not found: $CONFIG_PATH"
    
    # Try to create from example
    EXAMPLE_CONFIG="$SCRIPT_DIR/config/config_example.json"
    if [ -f "$EXAMPLE_CONFIG" ]; then
        log_info "Creating config from example..."
        cp "$EXAMPLE_CONFIG" "$CONFIG_PATH"
        log_warn "Please edit $CONFIG_PATH with your settings"
        exit 1
    else
        log_error "No config template found!"
        exit 1
    fi
fi

# Extract config values
DEVICE_ID=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('device_id', 'unknown'))")
BROKER_URL=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('broker_url', ''))")
RTSP_URL=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('camera', {}).get('rtsp_url', ''))" 2>/dev/null || echo "")

if [ "$QUICK_MODE" = false ]; then
    log_success "Config loaded"
    log_info "  Device ID: $DEVICE_ID"
    log_info "  Broker: $BROKER_URL"
    log_info "  Camera: ${RTSP_URL:+configured}${RTSP_URL:-NOT CONFIGURED}"
fi

#-------------------------------------------------------------------------------
# Step 4: Camera Discovery (if needed)
#-------------------------------------------------------------------------------

if [ "$QUICK_MODE" = false ]; then
    log_header "Step 4/6: Camera Setup"
fi

if [ "$DISCOVER_CAMERA" = true ] || [ -z "$RTSP_URL" ]; then
    log_info "Running camera discovery..."
    
    cd "$SCRIPT_DIR"
    if python3 scripts/discover_camera.py --config "$CONFIG_PATH"; then
        log_success "Camera configured"
        save_status "camera" "discovered"
        
        # Reload RTSP URL
        RTSP_URL=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('camera', {}).get('rtsp_url', ''))")
    else
        log_error "Camera discovery failed!"
        save_status "camera" "not_found"
        
        if [ -z "$RTSP_URL" ]; then
            log_error "No camera configured. Cannot continue."
            exit 1
        else
            log_warn "Using existing camera config: ${RTSP_URL##*@}"
        fi
    fi
elif [ "$QUICK_MODE" = false ]; then
    log_success "Camera already configured: ${RTSP_URL##*@}"
fi

#-------------------------------------------------------------------------------
# Step 5: IoT Broker Registration
#-------------------------------------------------------------------------------

if [ "$QUICK_MODE" = false ]; then
    log_header "Step 5/6: IoT Broker Registration"
fi

if [ "$OFFLINE_MODE" = true ]; then
    log_warn "Offline mode - skipping broker registration"
    save_status "broker_registered" "offline"
else
    if [ -z "$BROKER_URL" ]; then
        log_warn "No broker URL configured - running in offline mode"
        save_status "broker_registered" "no_broker"
    else
        log_info "Checking IoT broker and registering device..."
        
        cd "$SCRIPT_DIR"
        
        # Run registration check using dedicated script
        REGISTRATION_OUTPUT=$(python3 scripts/check_broker.py --register 2>&1)
        REGISTRATION_STATUS=$(echo "$REGISTRATION_OUTPUT" | grep "BROKER_STATUS=" | cut -d= -f2)
        
        # Show output
        echo "$REGISTRATION_OUTPUT" | grep -v "BROKER_STATUS=" | while read line; do
            if [[ "$line" == *"[OK]"* ]]; then
                log_success "${line#*] }"
            elif [[ "$line" == *"[WARN]"* ]]; then
                log_warn "${line#*] }"
            elif [[ "$line" == *"[ERROR]"* ]]; then
                log_error "${line#*] }"
            elif [[ "$line" == *"[INFO]"* ]]; then
                log_info "${line#*] }"
            fi
        done
        
        case "$REGISTRATION_STATUS" in
            registered|already_registered)
                save_status "broker_registered" "true"
                ;;
            unavailable)
                save_status "broker_registered" "unavailable"
                log_warn "Will retry registration during operation"
                ;;
            *)
                save_status "broker_registered" "false"
                ;;
        esac
    fi
fi

#-------------------------------------------------------------------------------
# Step 6: Launch Main Pipeline
#-------------------------------------------------------------------------------

log_header "Step 6/6: Starting Facial Recognition Service"

# Create logs directory
mkdir -p "$LOG_DIR"

# Show status summary
echo ""
echo "  Device ID:    $DEVICE_ID"
echo "  Camera:       ${RTSP_URL##*@}"
echo "  Broker:       $([ -n "$BROKER_URL" ] && echo "${BROKER_URL}" || echo "N/A")"
echo "  Registered:   $(get_status broker_registered)"
echo "  Debug:        $DEBUG_MODE"
echo ""

# Build command
CMD="python src/main.py --config $CONFIG_PATH"

if [ "$DEBUG_MODE" = true ]; then
    CMD="$CMD --debug"
fi

# Save startup status
save_status "last_start" "$(date -Iseconds)"
save_status "running" "true"

log_info "Launching pipeline..."
echo "============================================================"
echo ""

# Run with proper signal handling
cd "$SCRIPT_DIR"

cleanup() {
    echo ""
    log_info "Shutting down..."
    save_status "running" "false"
    save_status "last_stop" "$(date -Iseconds)"
}

trap cleanup EXIT INT TERM

# Execute
exec $CMD

