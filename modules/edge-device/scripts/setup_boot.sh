#!/bin/bash
#===============================================================================
# QRaie Edge Device - Boot Hardening Script
#===============================================================================
#
# Ensures the device boots into a fully operable state:
#   1. SSH server is enabled and starts on boot
#   2. QRaie facial recognition service starts on boot
#   3. Health check daemon starts on boot
#   4. Periodic health check timer is enabled
#   5. Network is configured for auto-start
#   6. Boot target is multi-user (CLI, no GUI overhead)
#   7. Hardware watchdog is configured (optional)
#
# After running this script, a reboot will bring the device up with:
#   - SSH accessible for remote management
#   - Facial recognition pipeline running
#   - Health monitoring active and self-healing
#
# Usage:
#   sudo ./setup_boot.sh                    # Full setup
#   sudo ./setup_boot.sh --check            # Verify current boot config
#   sudo ./setup_boot.sh --skip-watchdog    # Skip hardware watchdog setup
#
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="/opt/qraie"
SERVICE_USER="qraie"

# Flags
CHECK_ONLY=false
SKIP_WATCHDOG=false

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[ OK ]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; }

#===============================================================================
# Parse Arguments
#===============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --skip-watchdog)
            SKIP_WATCHDOG=true
            shift
            ;;
        --help|-h)
            echo "Usage: sudo $0 [--check] [--skip-watchdog]"
            echo ""
            echo "  --check          Only verify current boot configuration"
            echo "  --skip-watchdog  Skip hardware watchdog configuration"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check root (unless just checking)
if [ "$CHECK_ONLY" = false ] && [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Run as root (sudo)${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo "  QRaie Edge Device - Boot Configuration"
echo "============================================================"
echo "  Date: $(date)"
echo "  Mode: $([ "$CHECK_ONLY" = true ] && echo "Check Only" || echo "Setup")"
echo "============================================================"
echo ""

ISSUES=0
FIXED=0

#===============================================================================
# 1. SSH Server
#===============================================================================

echo -e "${BLUE}[1/7] SSH Server${NC}"

# Check if SSH is installed
if command -v sshd &>/dev/null || dpkg -s openssh-server &>/dev/null 2>&1; then
    log_success "SSH server is installed"
else
    log_error "SSH server is NOT installed"
    ISSUES=$((ISSUES + 1))
    if [ "$CHECK_ONLY" = false ]; then
        log_info "Installing openssh-server..."
        apt-get update -qq && apt-get install -y openssh-server
        FIXED=$((FIXED + 1))
    fi
fi

# Check if SSH is enabled at boot
SSH_SERVICE="ssh"
if ! systemctl is-enabled "$SSH_SERVICE" &>/dev/null 2>&1; then
    SSH_SERVICE="sshd"
fi

if systemctl is-enabled "$SSH_SERVICE" &>/dev/null 2>&1; then
    SSH_ENABLED=$(systemctl is-enabled "$SSH_SERVICE" 2>/dev/null)
    if [ "$SSH_ENABLED" = "enabled" ]; then
        log_success "SSH ($SSH_SERVICE) enabled at boot"
    else
        log_error "SSH ($SSH_SERVICE) is NOT enabled at boot (state: $SSH_ENABLED)"
        ISSUES=$((ISSUES + 1))
        if [ "$CHECK_ONLY" = false ]; then
            systemctl enable "$SSH_SERVICE"
            FIXED=$((FIXED + 1))
            log_success "SSH enabled at boot"
        fi
    fi
else
    log_error "SSH service not found"
    ISSUES=$((ISSUES + 1))
fi

# Check if SSH is currently running
if systemctl is-active "$SSH_SERVICE" &>/dev/null 2>&1; then
    log_success "SSH is currently running"
else
    log_warn "SSH is NOT currently running"
    ISSUES=$((ISSUES + 1))
    if [ "$CHECK_ONLY" = false ]; then
        systemctl start "$SSH_SERVICE"
        FIXED=$((FIXED + 1))
        log_success "SSH started"
    fi
fi

# Ensure SSH config allows password auth (for recovery)
SSHD_CONFIG="/etc/ssh/sshd_config"
if [ -f "$SSHD_CONFIG" ]; then
    if grep -q "^PasswordAuthentication yes" "$SSHD_CONFIG"; then
        log_success "SSH password authentication enabled"
    elif grep -q "^PasswordAuthentication no" "$SSHD_CONFIG"; then
        log_warn "SSH password authentication is disabled"
        if [ "$CHECK_ONLY" = false ]; then
            # Don't change this - just warn. Key-based auth is more secure.
            log_info "  (Key-based auth is more secure; leaving as-is)"
            log_info "  To enable: sudo sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' $SSHD_CONFIG"
        fi
    else
        log_success "SSH password authentication: using default (enabled)"
    fi
fi

echo ""

#===============================================================================
# 2. Boot Target (multi-user / CLI)
#===============================================================================

echo -e "${BLUE}[2/7] Boot Target${NC}"

CURRENT_TARGET=$(systemctl get-default 2>/dev/null || echo "unknown")
if [ "$CURRENT_TARGET" = "multi-user.target" ]; then
    log_success "Boot target: multi-user.target (CLI - optimal)"
elif [ "$CURRENT_TARGET" = "graphical.target" ]; then
    log_warn "Boot target: graphical.target (GUI wastes resources)"
    ISSUES=$((ISSUES + 1))
    if [ "$CHECK_ONLY" = false ]; then
        systemctl set-default multi-user.target
        FIXED=$((FIXED + 1))
        log_success "Changed to multi-user.target (reboot required)"
    fi
else
    log_info "Boot target: $CURRENT_TARGET"
fi

echo ""

#===============================================================================
# 3. QRaie Facial Recognition Service
#===============================================================================

echo -e "${BLUE}[3/7] QRaie Facial Recognition Service${NC}"

if [ -f "/etc/systemd/system/qraie-facial.service" ]; then
    log_success "Service file installed"

    FACIAL_ENABLED=$(systemctl is-enabled qraie-facial 2>/dev/null || echo "not-found")
    if [ "$FACIAL_ENABLED" = "enabled" ]; then
        log_success "qraie-facial enabled at boot"
    else
        log_error "qraie-facial NOT enabled at boot (state: $FACIAL_ENABLED)"
        ISSUES=$((ISSUES + 1))
        if [ "$CHECK_ONLY" = false ]; then
            systemctl enable qraie-facial
            FIXED=$((FIXED + 1))
            log_success "qraie-facial enabled at boot"
        fi
    fi

    FACIAL_ACTIVE=$(systemctl is-active qraie-facial 2>/dev/null || echo "unknown")
    if [ "$FACIAL_ACTIVE" = "active" ]; then
        log_success "qraie-facial is currently running"
    else
        log_warn "qraie-facial is NOT running (state: $FACIAL_ACTIVE)"
        if [ "$CHECK_ONLY" = false ]; then
            log_info "Starting qraie-facial..."
            systemctl start qraie-facial || log_warn "Start failed (may need config)"
        fi
    fi
else
    log_error "Service file not installed at /etc/systemd/system/qraie-facial.service"
    ISSUES=$((ISSUES + 1))
    if [ "$CHECK_ONLY" = false ]; then
        if [ -f "$PROJECT_DIR/systemd/qraie-facial.service" ]; then
            cp "$PROJECT_DIR/systemd/qraie-facial.service" /etc/systemd/system/
            systemctl daemon-reload
            systemctl enable qraie-facial
            FIXED=$((FIXED + 1))
            log_success "Installed and enabled qraie-facial service"
        else
            log_error "Source service file not found at $PROJECT_DIR/systemd/qraie-facial.service"
        fi
    fi
fi

echo ""

#===============================================================================
# 4. Health Check Daemon
#===============================================================================

echo -e "${BLUE}[4/7] Health Check Daemon${NC}"

if [ -f "/etc/systemd/system/qraie-health.service" ]; then
    log_success "Health daemon service file installed"

    HEALTH_ENABLED=$(systemctl is-enabled qraie-health 2>/dev/null || echo "not-found")
    if [ "$HEALTH_ENABLED" = "enabled" ]; then
        log_success "qraie-health enabled at boot"
    else
        log_warn "qraie-health NOT enabled at boot"
        ISSUES=$((ISSUES + 1))
        if [ "$CHECK_ONLY" = false ]; then
            systemctl enable qraie-health
            FIXED=$((FIXED + 1))
            log_success "qraie-health enabled at boot"
        fi
    fi
else
    log_error "Health daemon not installed"
    ISSUES=$((ISSUES + 1))
    if [ "$CHECK_ONLY" = false ]; then
        if [ -f "$PROJECT_DIR/systemd/qraie-health.service" ]; then
            cp "$PROJECT_DIR/systemd/qraie-health.service" /etc/systemd/system/
            systemctl daemon-reload
            systemctl enable qraie-health
            FIXED=$((FIXED + 1))
            log_success "Installed and enabled health daemon"
        fi
    fi
fi

echo ""

#===============================================================================
# 5. Health Check Timer (periodic one-shot)
#===============================================================================

echo -e "${BLUE}[5/7] Health Check Timer${NC}"

if [ -f "/etc/systemd/system/qraie-health-check.timer" ]; then
    log_success "Health check timer installed"

    TIMER_ENABLED=$(systemctl is-enabled qraie-health-check.timer 2>/dev/null || echo "not-found")
    if [ "$TIMER_ENABLED" = "enabled" ]; then
        log_success "Health check timer enabled at boot"
    else
        log_warn "Health check timer NOT enabled"
        ISSUES=$((ISSUES + 1))
        if [ "$CHECK_ONLY" = false ]; then
            systemctl enable qraie-health-check.timer
            systemctl start qraie-health-check.timer
            FIXED=$((FIXED + 1))
            log_success "Health check timer enabled and started"
        fi
    fi
else
    log_error "Health check timer not installed"
    ISSUES=$((ISSUES + 1))
    if [ "$CHECK_ONLY" = false ]; then
        for f in qraie-health-check.service qraie-health-check.timer; do
            if [ -f "$PROJECT_DIR/systemd/$f" ]; then
                cp "$PROJECT_DIR/systemd/$f" /etc/systemd/system/
            fi
        done
        systemctl daemon-reload
        systemctl enable qraie-health-check.timer
        systemctl start qraie-health-check.timer
        FIXED=$((FIXED + 1))
        log_success "Installed and enabled health check timer"
    fi
fi

echo ""

#===============================================================================
# 6. Network Configuration
#===============================================================================

echo -e "${BLUE}[6/7] Network Configuration${NC}"

# Check NetworkManager
if systemctl is-active NetworkManager &>/dev/null 2>&1; then
    log_success "NetworkManager is running"
elif systemctl is-active systemd-networkd &>/dev/null 2>&1; then
    log_success "systemd-networkd is running"
else
    log_warn "No network manager detected as active"
    ISSUES=$((ISSUES + 1))
fi

# Check if we have an IP address
IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$IP_ADDR" ]; then
    log_success "IP address: $IP_ADDR"
else
    log_error "No IP address detected"
    ISSUES=$((ISSUES + 1))
fi

# Check DNS
if host google.com &>/dev/null 2>&1 || nslookup google.com &>/dev/null 2>&1; then
    log_success "DNS resolution working"
else
    log_warn "DNS resolution may not be working"
fi

echo ""

#===============================================================================
# 7. Hardware Watchdog (optional)
#===============================================================================

echo -e "${BLUE}[7/7] Hardware Watchdog${NC}"

if [ "$SKIP_WATCHDOG" = true ]; then
    log_info "Hardware watchdog setup skipped (--skip-watchdog)"
else
    # Check if hardware watchdog is available (common on Jetson)
    if [ -e "/dev/watchdog" ] || [ -e "/dev/watchdog0" ]; then
        log_success "Hardware watchdog device available"

        # Check if systemd is configured to use it
        RUNTIME_WD=$(systemctl show -p RuntimeWatchdogUSec 2>/dev/null | cut -d= -f2)
        if [ "$RUNTIME_WD" != "0" ] && [ -n "$RUNTIME_WD" ] && [ "$RUNTIME_WD" != "infinity" ]; then
            log_success "systemd runtime watchdog active: $RUNTIME_WD"
        else
            log_warn "systemd runtime watchdog not configured"
            if [ "$CHECK_ONLY" = false ]; then
                # Configure systemd watchdog in system.conf
                SYSTEM_CONF="/etc/systemd/system.conf"
                if [ -f "$SYSTEM_CONF" ]; then
                    # Set RuntimeWatchdogSec=30 - if system hangs for 30s, it reboots
                    if ! grep -q "^RuntimeWatchdogSec=" "$SYSTEM_CONF"; then
                        echo "" >> "$SYSTEM_CONF"
                        echo "# QRaie: Hardware watchdog - reboot on 30s system hang" >> "$SYSTEM_CONF"
                        echo "RuntimeWatchdogSec=30" >> "$SYSTEM_CONF"
                        echo "RebootWatchdogSec=10min" >> "$SYSTEM_CONF"
                        FIXED=$((FIXED + 1))
                        log_success "Hardware watchdog configured (reboot required to activate)"
                    fi
                fi
            fi
        fi
    else
        log_info "No hardware watchdog device found (normal for some configurations)"
    fi
fi

echo ""

#===============================================================================
# Summary
#===============================================================================

echo "============================================================"
if [ "$CHECK_ONLY" = true ]; then
    echo "  Boot Configuration Check Complete"
else
    echo "  Boot Configuration Setup Complete"
fi
echo "============================================================"
echo ""

if [ $ISSUES -eq 0 ]; then
    echo -e "  ${GREEN}All checks passed - device is ready for reboot${NC}"
else
    if [ "$CHECK_ONLY" = true ]; then
        echo -e "  ${YELLOW}Found $ISSUES issue(s)${NC}"
        echo "  Run without --check to fix: sudo $0"
    else
        echo -e "  ${YELLOW}Found $ISSUES issue(s), fixed $FIXED${NC}"
        REMAINING=$((ISSUES - FIXED))
        if [ $REMAINING -gt 0 ]; then
            echo -e "  ${RED}$REMAINING issue(s) remain and may need manual attention${NC}"
        fi
    fi
fi

echo ""
echo "  On reboot, the device will start:"
echo "    1. SSH server (remote access)"
echo "    2. QRaie facial recognition pipeline"
echo "    3. Health check daemon (self-monitoring)"
echo "    4. Periodic health check timer (every 5 min)"
echo ""

# Show service status summary
echo "  Service Status:"
for svc in ssh qraie-facial qraie-health; do
    STATE=$(systemctl is-enabled "$svc" 2>/dev/null || echo "not-installed")
    ACTIVE=$(systemctl is-active "$svc" 2>/dev/null || echo "inactive")
    printf "    %-20s boot=%-10s now=%s\n" "$svc" "$STATE" "$ACTIVE"
done

TIMER_STATE=$(systemctl is-enabled qraie-health-check.timer 2>/dev/null || echo "not-installed")
printf "    %-20s boot=%-10s\n" "health-check.timer" "$TIMER_STATE"

echo ""
echo "============================================================"
echo ""

exit $ISSUES
