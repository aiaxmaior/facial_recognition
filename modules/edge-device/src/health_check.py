#!/usr/bin/env python3
"""
QRaie Edge Device - Self-Health Check System

Comprehensive health monitoring that:
- Checks all critical subsystems (CPU, memory, disk, temp, network, camera, pipeline, GPU)
- Performs automatic remediation where possible (restart service, clear temp files)
- Reports status to IoT broker via heartbeat
- Integrates with systemd watchdog for process-level monitoring
- Can run standalone or as a systemd timer target

Usage:
    # One-shot check (for systemd timer / cron)
    python health_check.py --config config/config.json

    # Continuous watchdog daemon
    python health_check.py --config config/config.json --daemon

    # JSON output for scripting
    python health_check.py --config config/config.json --json

Exit codes:
    0 = All healthy
    1 = Degraded (warnings, but operational)
    2 = Critical (action taken or required)
    3 = Fatal (device may need manual intervention)
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | health_check | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("health_check")


# ---------------------------------------------------------------------------
# Health status constants
# ---------------------------------------------------------------------------
STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_CRITICAL = "critical"
STATUS_UNKNOWN = "unknown"

EXIT_HEALTHY = 0
EXIT_DEGRADED = 1
EXIT_CRITICAL = 2
EXIT_FATAL = 3


# ---------------------------------------------------------------------------
# Individual health checks
# ---------------------------------------------------------------------------

def check_cpu(thresholds: dict) -> dict:
    """Check CPU load average."""
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().split()
        load_1, load_5, load_15 = float(parts[0]), float(parts[1]), float(parts[2])
        cpu_count = os.cpu_count() or 1
        usage_pct = (load_1 / cpu_count) * 100

        warn = thresholds.get("cpu_warn_percent", 85)
        crit = thresholds.get("cpu_crit_percent", 95)

        if usage_pct >= crit:
            status = STATUS_CRITICAL
        elif usage_pct >= warn:
            status = STATUS_DEGRADED
        else:
            status = STATUS_HEALTHY

        return {
            "name": "cpu",
            "status": status,
            "load_1m": load_1,
            "load_5m": load_5,
            "load_15m": load_15,
            "usage_percent": round(usage_pct, 1),
            "cpu_count": cpu_count,
        }
    except Exception as e:
        return {"name": "cpu", "status": STATUS_UNKNOWN, "error": str(e)}


def check_memory(thresholds: dict) -> dict:
    """Check memory utilisation."""
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    mem[parts[0].strip()] = int(parts[1].split()[0])

        total = mem.get("MemTotal", 1)
        available = mem.get("MemAvailable", 0)
        used_pct = (1 - available / total) * 100

        warn = thresholds.get("memory_warn_percent", 85)
        crit = thresholds.get("memory_crit_percent", 95)

        if used_pct >= crit:
            status = STATUS_CRITICAL
        elif used_pct >= warn:
            status = STATUS_DEGRADED
        else:
            status = STATUS_HEALTHY

        return {
            "name": "memory",
            "status": status,
            "total_mb": round(total / 1024, 1),
            "available_mb": round(available / 1024, 1),
            "used_percent": round(used_pct, 1),
        }
    except Exception as e:
        return {"name": "memory", "status": STATUS_UNKNOWN, "error": str(e)}


def check_disk(thresholds: dict) -> dict:
    """Check disk space on key mount points."""
    results = []
    paths_to_check = ["/", "/opt/qraie", "/tmp"]

    warn = thresholds.get("disk_warn_percent", 85)
    crit = thresholds.get("disk_crit_percent", 95)

    overall_status = STATUS_HEALTHY

    for path in paths_to_check:
        if not os.path.exists(path):
            continue
        try:
            usage = shutil.disk_usage(path)
            used_pct = (usage.used / usage.total) * 100

            if used_pct >= crit:
                s = STATUS_CRITICAL
            elif used_pct >= warn:
                s = STATUS_DEGRADED
            else:
                s = STATUS_HEALTHY

            if s == STATUS_CRITICAL:
                overall_status = STATUS_CRITICAL
            elif s == STATUS_DEGRADED and overall_status != STATUS_CRITICAL:
                overall_status = STATUS_DEGRADED

            results.append({
                "path": path,
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "free_gb": round(usage.free / (1024 ** 3), 2),
                "used_percent": round(used_pct, 1),
                "status": s,
            })
        except Exception as e:
            results.append({"path": path, "status": STATUS_UNKNOWN, "error": str(e)})

    return {"name": "disk", "status": overall_status, "mounts": results}


def check_temperature(thresholds: dict) -> dict:
    """Check Jetson thermal zones."""
    warn = thresholds.get("temp_warn_c", 75)
    crit = thresholds.get("temp_crit_c", 85)

    temps = []
    thermal_base = Path("/sys/devices/virtual/thermal")
    if not thermal_base.exists():
        return {"name": "temperature", "status": STATUS_UNKNOWN, "error": "no thermal zones found"}

    overall_status = STATUS_HEALTHY

    for zone_dir in sorted(thermal_base.glob("thermal_zone*")):
        try:
            temp_file = zone_dir / "temp"
            type_file = zone_dir / "type"
            temp_c = int(temp_file.read_text().strip()) / 1000.0
            zone_type = type_file.read_text().strip() if type_file.exists() else zone_dir.name

            if temp_c >= crit:
                s = STATUS_CRITICAL
            elif temp_c >= warn:
                s = STATUS_DEGRADED
            else:
                s = STATUS_HEALTHY

            if s == STATUS_CRITICAL:
                overall_status = STATUS_CRITICAL
            elif s == STATUS_DEGRADED and overall_status != STATUS_CRITICAL:
                overall_status = STATUS_DEGRADED

            temps.append({"zone": zone_type, "temp_c": round(temp_c, 1), "status": s})
        except Exception:
            pass

    if not temps:
        return {"name": "temperature", "status": STATUS_UNKNOWN, "error": "could not read any thermal zones"}

    max_temp = max(t["temp_c"] for t in temps)
    return {"name": "temperature", "status": overall_status, "max_c": max_temp, "zones": temps}


def check_network(config: dict) -> dict:
    """Check network connectivity (internet + broker)."""
    results = {}
    overall_status = STATUS_HEALTHY

    # 1. General internet connectivity (DNS resolution)
    try:
        socket.setdefaulttimeout(5)
        socket.getaddrinfo("8.8.8.8", 53)
        results["internet"] = "reachable"
    except Exception:
        results["internet"] = "unreachable"
        overall_status = STATUS_DEGRADED

    # 2. IoT broker reachability
    broker_url = config.get("broker_url", "")
    if broker_url:
        try:
            import requests
            resp = requests.get(f"{broker_url}/health", timeout=5, verify=True)
            results["broker"] = f"reachable (HTTP {resp.status_code})"
        except Exception as e:
            results["broker"] = f"unreachable ({e.__class__.__name__})"
            if overall_status != STATUS_CRITICAL:
                overall_status = STATUS_DEGRADED
    else:
        results["broker"] = "not_configured"

    # 3. Camera IP reachability (RTSP port 554)
    rtsp_url = config.get("camera", {}).get("rtsp_url", "")
    if rtsp_url:
        match = re.search(r"@([0-9.]+)", rtsp_url)
        if match:
            camera_ip = match.group(1)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((camera_ip, 554))
                sock.close()
                if result == 0:
                    results["camera_rtsp"] = f"reachable ({camera_ip}:554)"
                else:
                    results["camera_rtsp"] = f"unreachable ({camera_ip}:554)"
                    overall_status = STATUS_CRITICAL
            except Exception as e:
                results["camera_rtsp"] = f"error ({e})"
                overall_status = STATUS_CRITICAL
    else:
        results["camera_rtsp"] = "not_configured"

    return {"name": "network", "status": overall_status, **results}


def check_gpu() -> dict:
    """Check NVIDIA GPU health via nvidia-smi / tegrastats."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                temp = float(parts[0])
                util = float(parts[1])
                mem_used = float(parts[2])
                mem_total = float(parts[3])
                mem_pct = (mem_used / mem_total) * 100 if mem_total > 0 else 0

                status = STATUS_HEALTHY
                if temp > 85 or mem_pct > 95:
                    status = STATUS_CRITICAL
                elif temp > 75 or mem_pct > 85:
                    status = STATUS_DEGRADED

                return {
                    "name": "gpu",
                    "status": status,
                    "temp_c": temp,
                    "utilization_percent": util,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                    "memory_percent": round(mem_pct, 1),
                }
    except FileNotFoundError:
        pass
    except Exception as e:
        return {"name": "gpu", "status": STATUS_UNKNOWN, "error": str(e)}

    # Fallback: try reading Jetson GPU via /sys
    try:
        gpu_load_path = Path("/sys/devices/gpu.0/load")
        if gpu_load_path.exists():
            load = int(gpu_load_path.read_text().strip()) / 10.0
            return {"name": "gpu", "status": STATUS_HEALTHY, "utilization_percent": load}
    except Exception:
        pass

    return {"name": "gpu", "status": STATUS_UNKNOWN, "error": "nvidia-smi not available"}


def check_service(service_name: str = "qraie-facial") -> dict:
    """Check if the main facial recognition systemd service is running."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True, text=True, timeout=5,
        )
        state = result.stdout.strip()

        if state == "active":
            # Also check if the process is actually doing work
            pid_result = subprocess.run(
                ["systemctl", "show", service_name, "--property=MainPID", "--value"],
                capture_output=True, text=True, timeout=5,
            )
            pid = pid_result.stdout.strip()

            info = {"name": "service", "status": STATUS_HEALTHY, "state": state, "pid": pid}

            # Check process age (detect stuck restarts)
            if pid and pid != "0":
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text().split()
                    # Field 22 is start time in jiffies
                    uptime_s = time.time() - os.stat(f"/proc/{pid}").st_ctime
                    info["uptime_seconds"] = int(uptime_s)
                except Exception:
                    pass

            return info

        elif state == "activating":
            return {"name": "service", "status": STATUS_DEGRADED, "state": state}
        else:
            return {"name": "service", "status": STATUS_CRITICAL, "state": state}

    except FileNotFoundError:
        # systemctl not available (running outside systemd)
        return {"name": "service", "status": STATUS_UNKNOWN, "error": "systemctl not available"}
    except Exception as e:
        return {"name": "service", "status": STATUS_UNKNOWN, "error": str(e)}


def check_ssh() -> dict:
    """Check if SSH server is running and accessible."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "ssh"],
            capture_output=True, text=True, timeout=5,
        )
        state = result.stdout.strip()
        if state != "active":
            # Try sshd as well
            result = subprocess.run(
                ["systemctl", "is-active", "sshd"],
                capture_output=True, text=True, timeout=5,
            )
            state = result.stdout.strip()

        if state == "active":
            return {"name": "ssh", "status": STATUS_HEALTHY, "state": state}
        else:
            return {"name": "ssh", "status": STATUS_CRITICAL, "state": state}
    except Exception as e:
        return {"name": "ssh", "status": STATUS_UNKNOWN, "error": str(e)}


def check_recent_logs(log_dir: str = "logs") -> dict:
    """Check recent pipeline logs for error patterns."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return {"name": "logs", "status": STATUS_UNKNOWN, "error": "log directory not found"}

    # Find the most recent log file
    log_files = sorted(log_path.glob("pipeline_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        return {"name": "logs", "status": STATUS_UNKNOWN, "error": "no log files found"}

    latest = log_files[0]
    try:
        # Read last 200 lines
        with open(latest, "r") as f:
            lines = f.readlines()
        tail = lines[-200:] if len(lines) > 200 else lines

        errors = sum(1 for line in tail if "ERROR" in line)
        warnings = sum(1 for line in tail if "WARNING" in line or "WARN" in line)

        # Check for specific fatal patterns
        fatal_patterns = [
            "CUDA out of memory",
            "Segmentation fault",
            "killed",
            "Failed to initialize recognition",
            "All camera backends failed",
        ]
        fatal_found = []
        for pattern in fatal_patterns:
            for line in tail:
                if pattern.lower() in line.lower():
                    fatal_found.append(pattern)
                    break

        if fatal_found:
            status = STATUS_CRITICAL
        elif errors > 10:
            status = STATUS_DEGRADED
        else:
            status = STATUS_HEALTHY

        # Get log age
        age_seconds = time.time() - latest.stat().st_mtime

        return {
            "name": "logs",
            "status": status,
            "latest_log": str(latest),
            "log_age_seconds": int(age_seconds),
            "recent_errors": errors,
            "recent_warnings": warnings,
            "fatal_patterns": fatal_found,
        }
    except Exception as e:
        return {"name": "logs", "status": STATUS_UNKNOWN, "error": str(e)}


# ---------------------------------------------------------------------------
# Remediation actions
# ---------------------------------------------------------------------------

def remediate_disk(check_result: dict) -> List[str]:
    """Try to free disk space."""
    actions = []

    # Clear old log files (keep last 5)
    try:
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = sorted(log_dir.glob("pipeline_*.log"), key=lambda f: f.stat().st_mtime)
            if len(log_files) > 5:
                for old_log in log_files[:-5]:
                    old_log.unlink()
                    actions.append(f"Removed old log: {old_log.name}")
    except Exception as e:
        actions.append(f"Failed to clean logs: {e}")

    # Clear /tmp video buffer
    try:
        tmp_buffer = Path("/tmp/video_buffer")
        if tmp_buffer.exists():
            for f in tmp_buffer.iterdir():
                if f.is_file() and f.suffix in (".mp4", ".h265", ".h264"):
                    age = time.time() - f.stat().st_mtime
                    if age > 3600:  # Older than 1 hour
                        f.unlink()
                        actions.append(f"Removed stale video buffer: {f.name}")
    except Exception as e:
        actions.append(f"Failed to clean video buffer: {e}")

    # Clear Python cache
    try:
        for cache_dir in Path(".").rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
            actions.append(f"Removed __pycache__: {cache_dir}")
    except Exception:
        pass

    return actions


def remediate_service(check_result: dict) -> List[str]:
    """Try to restart the main service."""
    actions = []
    state = check_result.get("state", "unknown")

    if state in ("failed", "inactive", "dead"):
        try:
            result = subprocess.run(
                ["systemctl", "restart", "qraie-facial"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                actions.append("Restarted qraie-facial service")
            else:
                actions.append(f"Failed to restart service: {result.stderr.strip()}")
        except Exception as e:
            actions.append(f"Restart failed: {e}")

    return actions


def remediate_ssh(check_result: dict) -> List[str]:
    """Ensure SSH is running."""
    actions = []
    state = check_result.get("state", "unknown")

    if state != "active":
        for svc in ["ssh", "sshd"]:
            try:
                result = subprocess.run(
                    ["systemctl", "start", svc],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    actions.append(f"Started {svc} service")
                    # Also ensure it's enabled for boot
                    subprocess.run(
                        ["systemctl", "enable", svc],
                        capture_output=True, text=True, timeout=10,
                    )
                    actions.append(f"Enabled {svc} at boot")
                    break
            except Exception:
                continue

    return actions


# ---------------------------------------------------------------------------
# Main health check orchestrator
# ---------------------------------------------------------------------------

class HealthChecker:
    """Orchestrates all health checks and remediation."""

    def __init__(self, config: dict, config_path: str = None):
        self.config = config
        self.config_path = config_path
        self.health_config = config.get("health_check", {})
        self.thresholds = self.health_config.get("thresholds", {})
        self.auto_remediate = self.health_config.get("auto_remediate", True)
        self.device_id = config.get("device_id", "unknown")
        self._status_file = Path(config_path).parent.parent / ".health_status" if config_path else Path(".health_status")

    def run_all_checks(self) -> dict:
        """Run all health checks and return aggregated results."""
        start_time = time.time()

        checks = [
            check_cpu(self.thresholds),
            check_memory(self.thresholds),
            check_disk(self.thresholds),
            check_temperature(self.thresholds),
            check_network(self.config),
            check_gpu(),
            check_service(),
            check_ssh(),
            check_recent_logs(),
        ]

        # Determine overall status (worst of all checks)
        statuses = [c["status"] for c in checks]
        if STATUS_CRITICAL in statuses:
            overall = STATUS_CRITICAL
        elif STATUS_DEGRADED in statuses:
            overall = STATUS_DEGRADED
        elif STATUS_UNKNOWN in statuses:
            overall = STATUS_DEGRADED  # Unknown is a concern
        else:
            overall = STATUS_HEALTHY

        # Attempt remediation for critical/degraded checks
        remediations = []
        if self.auto_remediate:
            for check in checks:
                if check["status"] in (STATUS_CRITICAL, STATUS_DEGRADED):
                    actions = self._remediate(check)
                    if actions:
                        remediations.extend(actions)

        elapsed = time.time() - start_time

        report = {
            "device_id": self.device_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_status": overall,
            "checks": checks,
            "remediations": remediations,
            "check_duration_ms": round(elapsed * 1000, 1),
        }

        # Save status to file for other processes to read
        self._save_status(report)

        return report

    def _remediate(self, check: dict) -> List[str]:
        """Attempt remediation for a failed check."""
        name = check.get("name", "")
        actions = []

        try:
            if name == "disk" and check["status"] == STATUS_CRITICAL:
                actions = remediate_disk(check)
            elif name == "service" and check["status"] == STATUS_CRITICAL:
                actions = remediate_service(check)
            elif name == "ssh" and check["status"] == STATUS_CRITICAL:
                actions = remediate_ssh(check)
        except Exception as e:
            actions.append(f"Remediation error ({name}): {e}")

        return actions

    def _save_status(self, report: dict) -> None:
        """Save health status to a file for other services to read."""
        try:
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._status_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save health status: {e}")

    def send_health_heartbeat(self, report: dict) -> bool:
        """Send health report via heartbeat to IoT broker."""
        broker_url = self.config.get("broker_url", "")
        if not broker_url:
            return False

        try:
            import requests

            # Build a compact summary for heartbeat
            metrics = {}
            for check in report["checks"]:
                name = check["name"]
                if name == "cpu":
                    metrics["cpu_percent"] = check.get("usage_percent", 0)
                elif name == "memory":
                    metrics["memory_percent"] = check.get("used_percent", 0)
                elif name == "temperature":
                    metrics["temperature_c"] = check.get("max_c", 0)
                elif name == "disk":
                    for mount in check.get("mounts", []):
                        if mount["path"] == "/":
                            metrics["disk_percent"] = mount.get("used_percent", 0)
                            break
                elif name == "gpu":
                    metrics["gpu_temp_c"] = check.get("temp_c", 0)
                    metrics["gpu_memory_percent"] = check.get("memory_percent", 0)

            payload = {
                "timestamp": report["timestamp"],
                "status": report["overall_status"],
                "metrics": metrics,
                "health_checks": {c["name"]: c["status"] for c in report["checks"]},
                "remediations": report.get("remediations", []),
            }

            url = f"{broker_url}/data/devices/{self.device_id}/heartbeat"
            resp = requests.post(
                url,
                json=payload,
                timeout=10,
                headers={"X-Device-ID": self.device_id},
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Health heartbeat failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Daemon mode - continuous monitoring
# ---------------------------------------------------------------------------

def run_daemon(checker: HealthChecker, interval: int = 60):
    """Run health checks continuously as a daemon."""
    logger.info(f"Health check daemon started (interval={interval}s)")

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        logger.info(f"Received signal {sig}, shutting down")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while running:
        try:
            report = checker.run_all_checks()
            overall = report["overall_status"]
            check_count = len(report["checks"])
            remediation_count = len(report.get("remediations", []))

            log_fn = logger.info if overall == STATUS_HEALTHY else logger.warning
            log_fn(
                f"Health check: {overall.upper()} "
                f"({check_count} checks, {remediation_count} remediations, "
                f"{report['check_duration_ms']:.0f}ms)"
            )

            for check in report["checks"]:
                if check["status"] != STATUS_HEALTHY:
                    logger.warning(f"  [{check['status'].upper()}] {check['name']}: {json.dumps({k: v for k, v in check.items() if k not in ('name', 'status')}, default=str)}")

            for action in report.get("remediations", []):
                logger.info(f"  [REMEDIATION] {action}")

            # Send health status via heartbeat
            checker.send_health_heartbeat(report)

            # Notify systemd watchdog if running under it
            _notify_watchdog()

        except Exception as e:
            logger.error(f"Health check cycle failed: {e}")

        # Sleep in small intervals to allow signal handling
        for _ in range(interval):
            if not running:
                break
            time.sleep(1)

    logger.info("Health check daemon stopped")


def _notify_watchdog():
    """Notify systemd watchdog that we're still alive."""
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if not notify_socket:
        return

    try:
        if notify_socket.startswith("@"):
            notify_socket = "\0" + notify_socket[1:]

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(notify_socket)
        sock.sendall(b"WATCHDOG=1")
        sock.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QRaie Edge Device - Health Check")
    parser.add_argument(
        "--config", type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.json"),
        help="Path to device configuration file",
    )
    parser.add_argument("--daemon", action="store_true", help="Run as continuous watchdog daemon")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds (daemon mode)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--no-remediate", action="store_true", help="Disable automatic remediation")
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    else:
        logger.warning(f"Config not found: {args.config}, running with defaults")

    if args.no_remediate:
        config.setdefault("health_check", {})["auto_remediate"] = False

    checker = HealthChecker(config, config_path=args.config)

    if args.daemon:
        run_daemon(checker, interval=args.interval)
        return EXIT_HEALTHY

    # One-shot mode
    report = checker.run_all_checks()

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        # Human-readable output
        print(f"\n{'='*60}")
        print(f"  QRaie Health Check - {report['device_id']}")
        print(f"  {report['timestamp']}")
        print(f"{'='*60}\n")

        status_icons = {
            STATUS_HEALTHY: "[OK]",
            STATUS_DEGRADED: "[WARN]",
            STATUS_CRITICAL: "[CRIT]",
            STATUS_UNKNOWN: "[????]",
        }

        for check in report["checks"]:
            icon = status_icons.get(check["status"], "[????]")
            name = check["name"].upper().ljust(12)
            detail = ""

            if check["name"] == "cpu":
                detail = f"Load: {check.get('usage_percent', '?')}%"
            elif check["name"] == "memory":
                detail = f"Used: {check.get('used_percent', '?')}% ({check.get('available_mb', '?')} MB free)"
            elif check["name"] == "disk":
                mounts = check.get("mounts", [])
                parts = [f"{m['path']}={m.get('used_percent', '?')}%" for m in mounts]
                detail = ", ".join(parts)
            elif check["name"] == "temperature":
                detail = f"Max: {check.get('max_c', '?')}°C"
            elif check["name"] == "network":
                detail = f"Internet: {check.get('internet', '?')}, Broker: {check.get('broker', '?')}"
            elif check["name"] == "gpu":
                detail = f"Util: {check.get('utilization_percent', '?')}%, Mem: {check.get('memory_percent', '?')}%"
            elif check["name"] == "service":
                detail = f"State: {check.get('state', '?')}"
                if check.get("uptime_seconds"):
                    m, s = divmod(check["uptime_seconds"], 60)
                    h, m = divmod(m, 60)
                    detail += f", Up: {h:.0f}h{m:.0f}m"
            elif check["name"] == "ssh":
                detail = f"State: {check.get('state', '?')}"
            elif check["name"] == "logs":
                detail = f"Errors: {check.get('recent_errors', '?')}, Warnings: {check.get('recent_warnings', '?')}"
                if check.get("fatal_patterns"):
                    detail += f" FATAL: {check['fatal_patterns']}"

            print(f"  {icon:6s} {name} {detail}")

        if report.get("remediations"):
            print(f"\n  Remediations taken:")
            for action in report["remediations"]:
                print(f"    - {action}")

        overall = report["overall_status"].upper()
        print(f"\n  Overall: {overall} ({report['check_duration_ms']:.0f}ms)")
        print(f"{'='*60}\n")

    # Send heartbeat with health data
    checker.send_health_heartbeat(report)

    # Exit code based on overall status
    overall = report["overall_status"]
    if overall == STATUS_HEALTHY:
        return EXIT_HEALTHY
    elif overall == STATUS_DEGRADED:
        return EXIT_DEGRADED
    elif overall == STATUS_CRITICAL:
        return EXIT_CRITICAL
    else:
        return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())
