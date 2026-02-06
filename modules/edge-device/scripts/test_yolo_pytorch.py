#!/usr/bin/env python3
"""
Test YOLOv8-face detection using PyTorch directly (not TRT).
This helps verify if the issue is with the model or with TRT conversion.
"""
import cv2
import numpy as np
import time

# Capture a frame from RTSP
def capture_test_frame():
    """Capture a single frame from RTSP stream"""
    rtsp_url = "rtsp://admin:Fanatec2025@192.168.13.119/Preview_01_sub"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    for _ in range(10):  # Skip some frames to get a good one
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
    
    cap.release()
    return frame

def test_pytorch_yolo():
    """Test YOLOv8-face detection with PyTorch"""
    print("Capturing test frame from RTSP...")
    frame = capture_test_frame()
    if frame is None:
        print("Failed to capture frame, using test image")
        # Create a test pattern
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    
    print(f"Frame shape: {frame.shape}")
    cv2.imwrite("/tmp/test_frame_pytorch.jpg", frame)
    print("Saved frame to /tmp/test_frame_pytorch.jpg")
    
    # Load YOLOv8-face model
    model_path = "/home/qdrive/facial_recognition/modules/edge-device/models/yolov8_face/yolov8n-face.pt"
    print(f"Loading YOLOv8-face model: {model_path}")
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("Model loaded successfully")
        
        # Run inference
        print("Running inference...")
        start = time.time()
        results = model.predict(frame, conf=0.1, verbose=True)
        elapsed = time.time() - start
        print(f"Inference time: {elapsed*1000:.1f}ms")
        
        # Get detections
        for r in results:
            boxes = r.boxes
            print(f"Found {len(boxes)} detections")
            for box in boxes:
                conf = box.conf.cpu().numpy()[0]
                xyxy = box.xyxy.cpu().numpy()[0]
                print(f"  Detection: conf={conf:.4f}, box={xyxy}")
        
        # Save result
        annotated = results[0].plot()
        cv2.imwrite("/tmp/test_result_pytorch.jpg", annotated)
        print("Saved annotated result to /tmp/test_result_pytorch.jpg")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pytorch_yolo()
