#!/usr/bin/env python3
"""
Local Pipeline Test - Test camera + face detection + recognition without IoT broker

Usage:
    python scripts/test_pipeline_local.py
    python scripts/test_pipeline_local.py --duration 30
    python scripts/test_pipeline_local.py --display  # Show video window
"""

import argparse
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def connect_camera(rtsp_url: str) -> cv2.VideoCapture:
    """Connect to RTSP camera."""
    logger.info(f"Connecting to camera: {rtsp_url.split('@')[-1]}")  # Hide password
    
    # Try with FFmpeg backend first
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        logger.info("Connected via FFmpeg")
        return cap
    
    # Try default backend
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        logger.info("Connected via default backend")
        return cap
    
    logger.error("Failed to connect to camera")
    return None


def main():
    parser = argparse.ArgumentParser(description="Local pipeline test")
    parser.add_argument("--config", default="config/config.json", help="Config file path")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--display", action="store_true", help="Show video window")
    parser.add_argument("--process-fps", type=float, default=1.0, help="Frames to process per second")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / args.config
    
    # Load config
    config = load_config(str(config_path))
    camera_config = config.get("camera", {})
    recognition_config = config.get("recognition", {})
    
    rtsp_url = camera_config.get("rtsp_url")
    if not rtsp_url:
        logger.error("No RTSP URL in config")
        return 1
    
    # Load enrollments
    from iot_integration.db_manager import DatabaseManager
    db_path = config.get("sync", {}).get("enrollment_db_path", str(base_dir / "data/enrollments.db"))
    db = DatabaseManager(db_path)
    db.initialize()
    embeddings = db.get_all_embeddings()
    logger.info(f"Loaded {len(embeddings)} enrollments")
    
    # Initialize face recognition
    from deepface import DeepFace
    
    detector_backend = recognition_config.get("detector_backend", "yunet")
    model_name = recognition_config.get("model", "ArcFace")
    distance_threshold = recognition_config.get("distance_threshold", 0.45)
    min_confidence = recognition_config.get("min_confidence", 0.5)
    min_face_size = recognition_config.get("min_face_size", 40)
    
    logger.info(f"Detector: {detector_backend}, Model: {model_name}")
    logger.info(f"Distance threshold: {distance_threshold}, Min confidence: {min_confidence}")
    
    # Connect to camera
    cap = connect_camera(rtsp_url)
    if cap is None:
        return 1
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera: {width}x{height} @ {fps} FPS")
    
    # Processing loop
    print("\n" + "=" * 60)
    print("  Local Pipeline Test")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    frame_count = 0
    face_count = 0
    recognition_count = 0
    last_process_time = 0
    process_interval = 1.0 / args.process_fps
    
    try:
        while time.time() - start_time < args.duration:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Process at specified FPS
            if current_time - last_process_time < process_interval:
                if args.display:
                    cv2.imshow("Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            last_process_time = current_time
            
            # Detect faces
            try:
                faces = DeepFace.extract_faces(
                    frame,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    align=True
                )
                
                # Filter valid faces
                valid_faces = []
                for face in faces:
                    conf = face.get("confidence", 0)
                    if conf >= min_confidence:
                        region = face.get("facial_area", {})
                        w = region.get("w", 0)
                        h = region.get("h", 0)
                        if w >= min_face_size and h >= min_face_size:
                            valid_faces.append(face)
                
                if valid_faces:
                    face_count += len(valid_faces)
                    logger.info(f"[DETECT] Found {len(valid_faces)} face(s)")
                    
                    for i, face in enumerate(valid_faces):
                        conf = face.get("confidence", 0)
                        area = face.get("facial_area", {})
                        logger.info(f"  Face {i}: conf={conf:.3f}, area={area}")
                        
                        # Try recognition
                        face_img = face.get("face")
                        if face_img is not None:
                            # Convert to uint8 if needed
                            if face_img.dtype != np.uint8:
                                face_img = (face_img * 255).astype(np.uint8)
                            
                            try:
                                # Get embedding
                                result = DeepFace.represent(
                                    face_img,
                                    model_name=model_name,
                                    detector_backend="skip",
                                    enforce_detection=False
                                )
                                
                                if result:
                                    embedding = np.array(result[0]["embedding"], dtype=np.float32)
                                    
                                    # Compare against enrolled faces
                                    best_match = None
                                    best_distance = float('inf')
                                    
                                    for user_id, enrolled_emb in embeddings.items():
                                        distance = np.linalg.norm(embedding - enrolled_emb)
                                        if distance < best_distance:
                                            best_distance = distance
                                            best_match = user_id
                                    
                                    if best_distance <= distance_threshold:
                                        recognition_count += 1
                                        logger.info(f"  [MATCH] {best_match} (distance={best_distance:.4f})")
                                    else:
                                        logger.info(f"  [NO MATCH] Closest: {best_match} (distance={best_distance:.4f} > {distance_threshold})")
                                        
                            except Exception as e:
                                logger.warning(f"  Recognition error: {e}")
                        
                        # Draw on frame if display enabled
                        if args.display:
                            x, y = area.get("x", 0), area.get("y", 0)
                            w, h = area.get("w", 0), area.get("h", 0)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{conf:.2f}", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            except Exception as e:
                logger.warning(f"Detection error: {e}")
            
            # Display
            if args.display:
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Status every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                fps_actual = frame_count / elapsed
                logger.info(f"[STATUS] Frames: {frame_count}, FPS: {fps_actual:.1f}, "
                           f"Faces: {face_count}, Recognized: {recognition_count}")
    
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Frames read: {frame_count}")
    print(f"  Avg FPS: {frame_count/elapsed:.1f}")
    print(f"  Faces detected: {face_count}")
    print(f"  Recognitions: {recognition_count}")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
