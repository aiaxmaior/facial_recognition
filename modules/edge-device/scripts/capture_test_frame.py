#!/usr/bin/env python3
"""
Capture a test frame from the camera and run face detection on it.
Saves the frame to data/test_frame.jpg for inspection.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def main():
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config/config.json"
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    rtsp_url = config["camera"]["rtsp_url"]
    detector = config.get("recognition", {}).get("detector_backend", "yunet")
    
    # Connect to camera
    logger.info(f"Connecting to camera...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        logger.error("Failed to connect to camera")
        return 1
    
    # Read a few frames to let camera stabilize
    for _ in range(10):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Failed to read frame")
        return 1
    
    logger.info(f"Frame shape: {frame.shape}")
    
    # Save original frame
    output_path = base_dir / "data/test_frame.jpg"
    cv2.imwrite(str(output_path), frame)
    logger.info(f"Saved frame to: {output_path}")
    
    # Also save a resized version
    scale = 640 / max(frame.shape[:2])
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    resized_path = base_dir / "data/test_frame_resized.jpg"
    cv2.imwrite(str(resized_path), resized)
    logger.info(f"Saved resized frame ({resized.shape}) to: {resized_path}")
    
    # Run face detection on original
    logger.info(f"\nRunning face detection ({detector}) on original frame...")
    from deepface import DeepFace
    
    try:
        faces = DeepFace.extract_faces(
            frame,
            detector_backend=detector,
            enforce_detection=False,
            align=True
        )
        
        logger.info(f"Faces found: {len(faces)}")
        for i, face in enumerate(faces):
            conf = face.get("confidence", 0)
            area = face.get("facial_area", {})
            logger.info(f"  Face {i}: conf={conf:.3f}, area={area}")
            
    except Exception as e:
        logger.error(f"Detection error: {e}")
    
    # Run face detection on resized
    logger.info(f"\nRunning face detection ({detector}) on resized frame...")
    try:
        faces = DeepFace.extract_faces(
            resized,
            detector_backend=detector,
            enforce_detection=False,
            align=True
        )
        
        logger.info(f"Faces found: {len(faces)}")
        for i, face in enumerate(faces):
            conf = face.get("confidence", 0)
            area = face.get("facial_area", {})
            logger.info(f"  Face {i}: conf={conf:.3f}, area={area}")
            
    except Exception as e:
        logger.error(f"Detection error: {e}")
    
    print(f"\nCheck the captured frames at:")
    print(f"  {output_path}")
    print(f"  {resized_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
