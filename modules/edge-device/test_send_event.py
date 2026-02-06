#!/usr/bin/env python3
"""
Test script to send a face recognition event to the IoT broker.
"""
import sys
import cv2
import base64
import json
import requests
from datetime import datetime

# Add paths for imports
sys.path.insert(0, ".")
sys.path.insert(0, "iot_integration")

from iot_integration.schemas.event_schemas import FaceRecognitionEvent, FaceRecognitionMetadata
from iot_integration.image_utils import compress_image_for_event
from iot_integration.logging_config import setup_logging, build_debug_entries

# Configuration
BROKER_URL = "https://acetaxi-bridge.qryde.net/iot-broker/api"
DEVICE_ID = "jetson-001"
TEST_IMAGE_PATH = "data/test_frame.jpg"

# Person details
PERSON_NAME = "Arjun Joshi"
PERSON_ID = "arjun_joshi"


def register_device():
    """Register device with the IoT broker."""
    url = f"{BROKER_URL}/data/devices"
    
    payload = {
        "device_id": DEVICE_ID,
        "display_name": "Jetson Test Device",
        "capability": "face_recognition",
        "status": "provisioning",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    print(f"Registering device {DEVICE_ID}...")
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Device-ID": DEVICE_ID,
            },
            timeout=10
        )
        print(f"Registration status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code in (200, 201, 202, 409)  # 409 = already exists
    except requests.exceptions.RequestException as e:
        print(f"Registration error: {e}")
        return False


def main():
    # Set up structured logging
    setup_logging(
        device_id=DEVICE_ID,
        log_dir="logs",
        json_logs=True,
    )
    
    # First, try to register the device
    if not register_device():
        print("WARNING: Device registration failed, attempting to send event anyway...")
    
    print(f"\nLoading test image: {TEST_IMAGE_PATH}")
    
    # Load test image
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        print(f"ERROR: Could not load image from {TEST_IMAGE_PATH}")
        return 1
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Compress and encode image to base64
    print("Encoding image to base64...")
    image_b64 = compress_image_for_event(
        image,
        target_size_kb=50,
        initial_quality=65,
        crop_to_face=False,  # No face detection, use full image
        max_dimension=320
    )
    
    if image_b64:
        print(f"Image encoded: {len(image_b64)} characters (base64)")
    else:
        print("WARNING: Image encoding failed, sending without image")
    
    # Build debug entries for Graylog
    debug_entries = build_debug_entries(
        frame_id=12345,
        detection_time_ms=45.2,
        recognition_time_ms=23.1,
        faces_detected=1,
        pipeline_state="test",
        extra={"test_run": True, "source": "test_send_event.py"}
    )
    
    # Create event
    event = FaceRecognitionEvent(
        device_id=DEVICE_ID,
        person_name=PERSON_NAME,
        person_id=PERSON_ID,
        confidence=0.92,
        metadata=FaceRecognitionMetadata(
            confidence=0.92,
            person_detected=True,
            distance=0.35,
            frames_tracked=3,
            face_bbox=[100, 100, 150, 150]
        ),
        debug=debug_entries
    )
    
    # Get broker message format
    message = event.to_broker_message()
    
    # Add image to data payload
    if image_b64:
        message[1]["data"]["image"] = image_b64
    
    # Wrap in the expected payload structure
    payload = {
        "device_id": DEVICE_ID,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "messages": [message],
    }
    
    print("\n--- Event Payload ---")
    # Print without the full image for readability
    display_payload = json.loads(json.dumps(payload))
    if display_payload["messages"][0][1]["data"].get("image"):
        display_payload["messages"][0][1]["data"]["image"] = f"<base64 string, {len(image_b64)} chars>"
    print(json.dumps(display_payload, indent=2))
    
    # Send to broker
    url = f"{BROKER_URL}/data/events"
    print(f"\n--- Sending to {url} ---")
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Device-ID": DEVICE_ID,
            },
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code in (200, 201, 202):
            print("\nSUCCESS: Event accepted by broker!")
            return 0
        else:
            print(f"\nFAILED: Broker returned status {response.status_code}")
            return 1
            
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Request failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
