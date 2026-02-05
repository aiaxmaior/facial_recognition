#!/usr/bin/env python3
"""
Check IoT Broker Connection and Device Registration Status

Usage:
    python scripts/check_broker.py
    python scripts/check_broker.py --register  # Force registration
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Check IoT broker status")
    parser.add_argument("--config", default="config/config.json", help="Config file")
    parser.add_argument("--register", action="store_true", help="Attempt registration")
    parser.add_argument("--heartbeat", action="store_true", help="Send heartbeat")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / args.config
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    device_id = config.get("device_id", "unknown")
    broker_url = config.get("broker_url", "")
    
    print("=" * 50)
    print("  IoT Broker Status Check")
    print("=" * 50)
    print(f"  Device ID: {device_id}")
    print(f"  Broker URL: {broker_url}")
    print("=" * 50)
    
    if not broker_url:
        print("\n[ERROR] No broker URL configured")
        return 1
    
    from iot_integration.iot_client import IoTClient, IoTClientConfig
    
    iot_config = IoTClientConfig(
        device_id=device_id,
        broker_url=broker_url
    )
    
    client = IoTClient(iot_config)
    client.start()
    
    try:
        # Try to register
        if args.register:
            print("\n[INFO] Attempting device registration...")
            try:
                result = client.register_device()
                if result:
                    print("[OK] Device registered successfully")
                    print("BROKER_STATUS=registered")
                else:
                    print("[WARN] Registration returned false")
                    print("BROKER_STATUS=failed")
            except Exception as e:
                if "409" in str(e):
                    print("[OK] Device already registered (409 Conflict)")
                    print("BROKER_STATUS=already_registered")
                elif "503" in str(e):
                    print("[WARN] Broker unavailable (503 Service Unavailable)")
                    print("BROKER_STATUS=unavailable")
                else:
                    print(f"[ERROR] Registration error: {e}")
                    print("BROKER_STATUS=error")
        
        # Try heartbeat
        if args.heartbeat:
            print("\n[INFO] Sending heartbeat...")
            try:
                result = client.send_heartbeat()
                if result:
                    print("[OK] Heartbeat sent successfully")
                    print("HEARTBEAT_STATUS=ok")
                else:
                    print("[WARN] Heartbeat returned false")
                    print("HEARTBEAT_STATUS=failed")
            except Exception as e:
                print(f"[ERROR] Heartbeat error: {e}")
                print("HEARTBEAT_STATUS=error")
        
        # Just check connectivity
        if not args.register and not args.heartbeat:
            print("\n[INFO] Checking broker connectivity...")
            try:
                import requests
                response = requests.get(f"{broker_url}/health", timeout=5)
                if response.status_code == 200:
                    print("[OK] Broker is reachable")
                    print("BROKER_STATUS=online")
                else:
                    print(f"[WARN] Broker returned status {response.status_code}")
                    print("BROKER_STATUS=degraded")
            except requests.exceptions.ConnectionError:
                print("[ERROR] Cannot connect to broker")
                print("BROKER_STATUS=offline")
            except Exception as e:
                print(f"[WARN] Connection check: {e}")
                print("BROKER_STATUS=unknown")
        
        return 0
        
    finally:
        client.stop()


if __name__ == "__main__":
    sys.exit(main())
