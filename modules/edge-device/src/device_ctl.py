#!/usr/bin/env python3
"""
Device Control CLI - Register and test device connectivity.

Usage:
    python device_ctl.py register [--config CONFIG_PATH]
    python device_ctl.py heartbeat [--config CONFIG_PATH]
    python device_ctl.py status [--config CONFIG_PATH]
    python device_ctl.py camera [--config CONFIG_PATH]
"""

import argparse
import getpass
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iot_integration.iot_client import IoTClient, IoTClientConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def get_system_metrics() -> dict:
    """Get basic system metrics."""
    metrics = {}
    
    # CPU load
    try:
        with open('/proc/loadavg', 'r') as f:
            load = float(f.read().split()[0])
        metrics["cpu_percent"] = min(load * 100 / os.cpu_count(), 100.0)
    except Exception:
        metrics["cpu_percent"] = 0.0
    
    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split(':')
            if len(parts) == 2:
                mem_info[parts[0]] = int(parts[1].split()[0])
        total = mem_info.get('MemTotal', 1)
        available = mem_info.get('MemAvailable', 0)
        metrics["memory_percent"] = (1 - available / total) * 100
    except Exception:
        metrics["memory_percent"] = 0.0
    
    # Temperature (Jetson)
    try:
        with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
            metrics["temperature_c"] = int(f.read().strip()) / 1000.0
    except Exception:
        metrics["temperature_c"] = 0.0
    
    return metrics


def cmd_register(config: dict, args):
    """Register device with IoT broker."""
    iot_config = IoTClientConfig(
        device_id=config.get("device_id", "unknown-device"),
        broker_url=config.get("broker_url"),
    )
    
    client = IoTClient(iot_config)
    client.start()
    
    display_name = args.name or config.get("device_id")
    
    logger.info(f"Registering device: {iot_config.device_id}")
    logger.info(f"Broker URL: {iot_config.broker_url}")
    
    success = client.register_device(
        display_name=display_name,
        capability="face_recognition",
        status="provisioning",
    )
    
    client.stop()
    
    if success:
        logger.info("Device registered successfully!")
        return 0
    else:
        logger.error("Device registration failed")
        return 1


def cmd_heartbeat(config: dict, args):
    """Send heartbeat to IoT broker."""
    import time
    
    iot_config = IoTClientConfig(
        device_id=config.get("device_id", "unknown-device"),
        broker_url=config.get("broker_url"),
    )
    
    client = IoTClient(iot_config)
    client.start()
    
    interval = config.get("heartbeat", {}).get("interval_seconds", 30)
    loop_mode = getattr(args, 'loop', False)
    
    if loop_mode:
        logger.info(f"Starting continuous heartbeat for: {iot_config.device_id}")
        logger.info(f"Interval: {interval}s (Ctrl+C to stop)")
    
    try:
        while True:
            metrics = get_system_metrics()
            
            logger.info(f"Sending heartbeat for: {iot_config.device_id}")
            logger.info(f"Metrics: CPU={metrics['cpu_percent']:.1f}%, "
                        f"Memory={metrics['memory_percent']:.1f}%, "
                        f"Temp={metrics['temperature_c']:.1f}°C")
            
            success = client.send_heartbeat(metrics=metrics)
            
            if success:
                logger.info("Heartbeat sent successfully!")
            else:
                logger.error("Heartbeat failed")
            
            if not loop_mode:
                client.stop()
                return 0 if success else 1
            
            # Wait for next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("\nHeartbeat loop stopped by user")
        client.stop()
        return 0


def cmd_status(config: dict, args):
    """Show device configuration status."""
    print("\n=== Device Configuration ===")
    print(f"Device ID:  {config.get('device_id', 'NOT SET')}")
    print(f"Broker URL: {config.get('broker_url', 'NOT SET')}")
    
    camera = config.get('camera', {})
    rtsp = camera.get('rtsp_url', 'NOT SET')
    # Hide password in URL
    if '@' in rtsp:
        rtsp = rtsp.split('@')[-1]
        rtsp = f"rtsp://***@{rtsp}"
    print(f"Camera:     {rtsp}")
    
    print("\n=== System Metrics ===")
    metrics = get_system_metrics()
    print(f"CPU:        {metrics['cpu_percent']:.1f}%")
    print(f"Memory:     {metrics['memory_percent']:.1f}%")
    print(f"Temperature:{metrics['temperature_c']:.1f}°C")
    
    return 0


def scan_for_rtsp_cameras(subnet: str = None, timeout: float = 0.5) -> list:
    """
    Scan local network for devices with RTSP port 554 open.
    
    Args:
        subnet: Subnet to scan (e.g., "192.168.13"). Auto-detected if None.
        timeout: Connection timeout in seconds.
        
    Returns:
        List of IP addresses with port 554 open.
    """
    # Auto-detect subnet from default interface
    if subnet is None:
        try:
            # Get local IP by connecting to external address (doesn't actually connect)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            subnet = ".".join(local_ip.split(".")[:-1])
        except Exception:
            subnet = "192.168.1"
    
    logger.info(f"Scanning {subnet}.0/24 for RTSP cameras (port 554)...")
    
    found = []
    for i in range(1, 255):
        ip = f"{subnet}.{i}"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, 554))
            sock.close()
            if result == 0:
                found.append(ip)
                logger.info(f"  Found RTSP device: {ip}")
        except Exception:
            pass
    
    return found


def test_rtsp_stream(rtsp_url: str, duration: int = 5) -> bool:
    """
    Test RTSP stream using ffplay.
    
    Args:
        rtsp_url: Full RTSP URL to test.
        duration: How long to play (seconds).
        
    Returns:
        True if stream played successfully.
    """
    print(f"\nTesting camera stream for {duration} seconds...")
    print("(Close the ffplay window or wait for it to end)\n")
    
    try:
        # Run ffplay with timeout, hide most output
        result = subprocess.run(
            [
                "ffplay",
                "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-t", str(duration),
                "-autoexit",
                "-loglevel", "warning",
            ],
            timeout=duration + 10,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return True  # Played for duration, consider success
    except FileNotFoundError:
        logger.error("ffplay not found. Install ffmpeg: sudo apt install ffmpeg")
        return False
    except Exception as e:
        logger.error(f"Stream test failed: {e}")
        return False


def update_config_rtsp(config_path: str, rtsp_url: str) -> bool:
    """
    Update the RTSP URL in config file.
    
    Args:
        config_path: Path to config JSON file.
        rtsp_url: New RTSP URL to set.
        
    Returns:
        True if updated successfully.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Handle both flat and nested camera config
        if 'camera' in config:
            config['camera']['rtsp_url'] = rtsp_url
        else:
            config['rtsp_url'] = rtsp_url
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return True
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False


def cmd_camera(config: dict, args):
    """Auto-discover and configure camera."""
    print("\n" + "=" * 50)
    print("  QRaie Camera Auto-Configuration")
    print("=" * 50 + "\n")
    
    # Step 1: Scan for cameras
    print("Step 1: Scanning network for RTSP cameras...\n")
    cameras = scan_for_rtsp_cameras()
    
    if not cameras:
        print("\nNo RTSP cameras found on the network.")
        print("Please ensure the camera is:")
        print("  - Connected to the same network/switch")
        print("  - Powered on")
        print("  - Has RTSP enabled (port 554)")
        return 1
    
    # Step 2: Select camera if multiple found
    camera_ip = None
    if len(cameras) == 1:
        camera_ip = cameras[0]
        print(f"\nFound camera at: {camera_ip}")
    else:
        print(f"\nFound {len(cameras)} cameras:")
        for i, ip in enumerate(cameras, 1):
            print(f"  [{i}] {ip}")
        
        while camera_ip is None:
            try:
                choice = input("\nSelect camera number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return 1
                idx = int(choice) - 1
                if 0 <= idx < len(cameras):
                    camera_ip = cameras[idx]
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a number.")
    
    # Step 3: Get credentials
    print("\n" + "-" * 50)
    print("Step 2: Camera Credentials")
    print("-" * 50)
    print("\nEnter your Reolink camera credentials.")
    print("(Password will not be displayed)\n")
    
    username = input("Camera username: ").strip()
    if not username:
        username = "admin"
        print(f"  Using default: {username}")
    
    password = getpass.getpass("Camera password: ")
    if not password:
        print("Error: Password is required.")
        return 1
    
    # Step 4: Build and test RTSP URL
    # Reolink stream path - exact case matters!
    stream_path = "Preview_01_sub"
    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/{stream_path}"
    
    print("\n" + "-" * 50)
    print("Step 3: Testing Camera Stream")
    print("-" * 50)
    print(f"\nCamera IP: {camera_ip}")
    print(f"Stream: {stream_path}")
    
    success = test_rtsp_stream(rtsp_url, duration=args.test_duration if hasattr(args, 'test_duration') else 5)
    
    if not success:
        print("\nStream test failed. Please check:")
        print("  - Username and password are correct")
        print("  - Camera supports RTSP streaming")
        print("  - Stream path 'Preview_01_sub' is available")
        
        retry = input("\nRetry with different credentials? (y/n): ").strip().lower()
        if retry == 'y':
            return cmd_camera(config, args)
        return 1
    
    # Step 5: Save configuration
    print("\n" + "-" * 50)
    print("Step 4: Save Configuration")
    print("-" * 50)
    
    save = input(f"\nSave camera configuration to {args.config}? (y/n): ").strip().lower()
    if save == 'y':
        if update_config_rtsp(args.config, rtsp_url):
            print("\nConfiguration saved successfully!")
            print(f"Camera URL: rtsp://***@{camera_ip}/{stream_path}")
        else:
            print("\nFailed to save configuration.")
            return 1
    else:
        print("\nConfiguration not saved.")
        print(f"\nTo manually configure, set rtsp_url to:")
        print(f"  rtsp://{username}:****@{camera_ip}/{stream_path}")
    
    print("\n" + "=" * 50)
    print("  Camera configuration complete!")
    print("=" * 50 + "\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="QRaie Device Control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python device_ctl.py register            # Register device
  python device_ctl.py heartbeat           # Send single heartbeat
  python device_ctl.py heartbeat --loop    # Continuous heartbeat (30s)
  python device_ctl.py status              # Show status
  python device_ctl.py camera              # Auto-configure camera
        """
    )
    

    subparsers = parser.add_subparsers(dest="command", help="Commands")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "/home/qdrive/facial_recognition/modules/edge-device/config/config.json"),
        help="Path to device configuration file"
    )
    
    # register command
    reg_parser = subparsers.add_parser("register", help="Register device with broker")
    reg_parser.add_argument("--name", type=str, help="Display name for device")
    
    # heartbeat command
    hb_parser = subparsers.add_parser("heartbeat", help="Send heartbeat to broker")
    hb_parser.add_argument("--loop", action="store_true", 
                           help="Send heartbeat continuously (every 30s)")
    
    # status command
    subparsers.add_parser("status", help="Show device status")
    
    # camera command
    camera_parser = subparsers.add_parser("camera", help="Auto-discover and configure camera")
    camera_parser.add_argument("--test-duration", type=int, default=5, 
                               help="Duration to test stream (seconds)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Camera command can work without existing config
    if args.command == "camera":
        config = {}
        if os.path.exists(args.config):
            config = load_config(args.config)
        return cmd_camera(config, args)
    
    # Other commands need config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        logger.info("Create config at: /opt/qraie/config/device_config.json")
        return 1
    
    config = load_config(args.config)
    
    # Execute command
    if args.command == "register":
        return cmd_register(config, args)
    elif args.command == "heartbeat":
        return cmd_heartbeat(config, args)
    elif args.command == "status":
        return cmd_status(config, args)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
