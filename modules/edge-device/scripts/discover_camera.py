#!/usr/bin/env python3
"""
Camera Discovery Script
Automatically finds Reolink cameras on the network and updates config.
"""

import argparse
import json
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_local_subnets():
    """Get all local network subnets from active interfaces."""
    subnets = []
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if 'inet ' in line and '127.0.0.1' not in line:
                # Extract IP like 192.168.13.81/24
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part == 'inet' and i + 1 < len(parts):
                        ip_cidr = parts[i + 1]
                        ip = ip_cidr.split('/')[0]
                        # Get subnet base (e.g., 192.168.13)
                        subnet_base = '.'.join(ip.split('.')[:3])
                        subnets.append(subnet_base)
    except Exception as e:
        print(f"Error getting subnets: {e}")
    return subnets


def check_rtsp_port(ip, port=554, timeout=1):
    """Check if RTSP port is open on given IP."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False


def test_rtsp_stream(ip, username, password, stream_path, timeout=8):
    """Test if RTSP stream is accessible with given credentials."""
    url = f"rtsp://{username}:{password}@{ip}/{stream_path}"
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", url],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0 or '"codec_type": "video"' in result.stdout:
            return True, url
    except subprocess.TimeoutExpired:
        # Timeout often means stream is there but slow
        return True, url
    except Exception as e:
        pass
    return False, None


def scan_subnet(subnet_base, port=554):
    """Scan a subnet for devices with open RTSP port."""
    found = []
    
    def check_ip(i):
        ip = f"{subnet_base}.{i}"
        if check_rtsp_port(ip, port):
            return ip
        return None
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(check_ip, i): i for i in range(1, 255)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                found.append(result)
    
    return found


def discover_cameras(username="admin", password="Fanatec2025", 
                     stream_paths=["Preview_01_main", "Preview_01_sub"]):
    """Discover cameras on all local subnets."""
    print("Discovering local network subnets...")
    subnets = get_local_subnets()
    
    if not subnets:
        print("No network subnets found!")
        return []
    
    print(f"Found subnets: {subnets}")
    
    cameras = []
    for subnet in subnets:
        print(f"\nScanning {subnet}.0/24 for RTSP devices...")
        rtsp_hosts = scan_subnet(subnet)
        
        if rtsp_hosts:
            print(f"  Found {len(rtsp_hosts)} device(s) with RTSP port open: {rtsp_hosts}")
            
            for ip in rtsp_hosts:
                for stream_path in stream_paths:
                    print(f"  Testing {ip} with stream '{stream_path}'...")
                    success, url = test_rtsp_stream(ip, username, password, stream_path)
                    if success:
                        print(f"    ✓ Camera found: {ip} (stream: {stream_path})")
                        cameras.append({
                            "ip": ip,
                            "stream_path": stream_path,
                            "url": url
                        })
                        break  # Found working stream, move to next IP
        else:
            print(f"  No RTSP devices found on {subnet}.0/24")
    
    return cameras


def update_config(config_path, camera_url):
    """Update config file with new camera URL."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        old_url = config.get('camera', {}).get('rtsp_url', 'N/A')
        config['camera']['rtsp_url'] = camera_url
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\nConfig updated:")
        print(f"  Old: {old_url}")
        print(f"  New: {camera_url}")
        return True
    except Exception as e:
        print(f"Error updating config: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Discover cameras and update config")
    parser.add_argument("--config", "-c", 
                        default="config/config.json",
                        help="Path to config file")
    parser.add_argument("--username", "-u", 
                        default="admin",
                        help="Camera username")
    parser.add_argument("--password", "-p", 
                        default="Fanatec2025",
                        help="Camera password")
    parser.add_argument("--stream", "-s",
                        default="Preview_01_main",
                        help="Preferred stream path")
    parser.add_argument("--dry-run", "-n",
                        action="store_true",
                        help="Don't update config, just show what would be done")
    parser.add_argument("--subnet",
                        help="Specific subnet to scan (e.g., 192.168.13)")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("  Camera Discovery Tool")
    print("=" * 50)
    
    # Discover cameras
    if args.subnet:
        subnets = [args.subnet]
        print(f"\nScanning specified subnet: {args.subnet}.0/24")
        rtsp_hosts = scan_subnet(args.subnet)
        cameras = []
        for ip in rtsp_hosts:
            success, url = test_rtsp_stream(ip, args.username, args.password, args.stream)
            if success:
                cameras.append({"ip": ip, "stream_path": args.stream, "url": url})
    else:
        cameras = discover_cameras(
            username=args.username,
            password=args.password,
            stream_paths=[args.stream, "Preview_01_sub", "Preview_01_main"]
        )
    
    if not cameras:
        print("\n✗ No cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check camera is powered on")
        print("  2. Check camera is connected to the same network")
        print("  3. Verify credentials are correct")
        print("  4. Try specifying subnet: --subnet 192.168.x")
        sys.exit(1)
    
    print(f"\n{'=' * 50}")
    print(f"  Found {len(cameras)} camera(s)")
    print(f"{'=' * 50}")
    
    for i, cam in enumerate(cameras, 1):
        print(f"  [{i}] {cam['ip']} - {cam['stream_path']}")
    
    # Use first camera found
    selected = cameras[0]
    camera_url = selected['url']
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would update config with: {camera_url}")
    else:
        update_config(args.config, camera_url)
    
    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
