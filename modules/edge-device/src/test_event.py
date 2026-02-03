#!/usr/bin/env python3
"""
Test Event Sender - Send mock recognition events to IoT broker.

Usage:
    python test_event.py                       # Send basic test event
    python test_event.py --low-confidence      # Send low-confidence test event
    python test_event.py --image face.jpg      # Include image in payload
    python test_event.py --user "EMP-001"      # Specify user ID
    python test_event.py --name "John Doe"     # Specify person name

NOTE: The API does not support 'Unknown' as person_name.
      Unrecognized faces are logged locally but not sent to the broker.
"""

import argparse
import base64
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iot_integration.iot_client import IoTClient, IoTClientConfig
from iot_integration.schemas.event_schemas import FaceRecognitionEvent

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
        return {}


def load_image_base64(image_path: str) -> str:
    """Load image and encode as base64."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Determine mime type
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
        }
        mime = mime_types.get(ext, 'image/jpeg')
        
        # Encode as data URI
        b64 = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime};base64,{b64}"
        
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None


def send_test_event(config: dict, args) -> bool:
    """Send a test recognition event."""
    
    device_id = config.get("device_id", "test-device")
    broker_url = config.get("broker_url")
    
    if not broker_url:
        logger.error("No broker_url configured!")
        return False
    
    # Create IoT client
    iot_config = IoTClientConfig(
        device_id=device_id,
        broker_url=broker_url,
    )
    client = IoTClient(iot_config)
    client.start()
    
    # Determine event type
    is_low_confidence = getattr(args, 'low_confidence', False)
    
    if is_low_confidence:
        # Build low-confidence recognition event (for testing edge cases)
        # NOTE: API spec says person_name cannot be "Unknown" - using test user instead
        event = FaceRecognitionEvent(
            device_id=device_id,
            person_name="Test_LowConfidence",
            person_id="TEST-LOW-CONF",
            confidence=0.35,  # Below typical threshold
            timestamp=datetime.utcnow(),
            metadata={
                "test_type": "low_confidence",
                "message": "TEST: Simulated low-confidence recognition",
                "bbox": [100, 100, 150, 150],
            },
            debug=[{
                "level": "info",
                "message": "TEST: Low confidence recognition event",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "test_mode": True,
            }]
        )
        event_type = "Low Confidence Test"
    else:
        # Build normal recognition event
        event = FaceRecognitionEvent(
            device_id=device_id,
            person_name=args.name,
            person_id=args.user,
            confidence=args.confidence,
            timestamp=datetime.utcnow(),
        )
        event_type = "Recognition"
    
    # Add image if provided
    image_b64 = None
    if args.image and os.path.exists(args.image):
        logger.info(f"Loading image: {args.image}")
        image_b64 = load_image_base64(args.image)
        if image_b64:
            logger.info(f"Image encoded: {len(image_b64)} chars")
    
    # Display event info
    print("\n" + "=" * 50)
    print(f"  Test {event_type} Event")
    print("=" * 50)
    print(f"  Device ID:   {device_id}")
    print(f"  Event Type:  {event_type}")
    if is_low_confidence:
        print(f"  Person ID:   TEST-LOW-CONF")
        print(f"  Person Name: Test_LowConfidence")
        print(f"  Confidence:  0.35 (below threshold)")
        print(f"  Test Type:   Low confidence simulation")
    else:
        print(f"  Person ID:   {args.user}")
        print(f"  Person Name: {args.name}")
        print(f"  Confidence:  {args.confidence}")
    print(f"  Image:       {'Yes' if image_b64 else 'No'}")
    print(f"  Broker:      {broker_url}")
    print("=" * 50 + "\n")
    
    # Send event
    logger.info(f"Sending test {event_type.lower()} event...")
    
    try:
        # Use sync method for immediate feedback
        success = client.send_event_sync(event)
        
        if success:
            logger.info("Event sent successfully!")
            print(f"\n[SUCCESS] {event_type} event received by broker\n")
        else:
            logger.error("Event transmission failed")
            print(f"\n[FAILED] {event_type} event not received by broker\n")
        
        client.stop()
        return success
        
    except Exception as e:
        logger.error(f"Error sending event: {e}")
        client.stop()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Send test recognition events to IoT broker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_event.py                              # Basic recognition test
  python test_event.py --low-confidence             # Low-confidence test
  python test_event.py --image test_face.jpg       # Recognition with image
  python test_event.py --user EMP-001 --name "John Doe"
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.json"),
        help="Path to config file"
    )
    parser.add_argument(
        "--low-confidence",
        action="store_true",
        dest="low_confidence",
        help="Send low-confidence test event (for testing threshold handling)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="TEST-001",
        help="Person/Employee ID (default: TEST-001)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Test User",
        help="Person display name (default: Test User)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.92,
        help="Recognition confidence 0-1 (default: 0.92)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to face image (jpg/png) to include in payload"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if not config:
        logger.error(f"Could not load config from: {args.config}")
        return 1
    
    # Send test event
    success = send_test_event(config, args)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
