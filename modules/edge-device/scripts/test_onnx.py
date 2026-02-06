#!/usr/bin/env python3
"""
Test ONNX model inference to verify the export is correct.
"""
import cv2
import numpy as np
import time
import onnxruntime as ort

def capture_test_frame():
    """Capture a single frame from RTSP stream"""
    rtsp_url = "rtsp://admin:Fanatec2025@192.168.13.119/Preview_01_sub"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    for _ in range(10):  # Skip some frames
        ret, frame = cap.read()
        if not ret:
            return None
    
    cap.release()
    return frame

def preprocess(frame, input_size=640):
    """Preprocess frame for YOLOv8"""
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = normalized.transpose(2, 0, 1)
    return np.ascontiguousarray(chw[np.newaxis, ...]), scale, (pad_x, pad_y)

def test_onnx():
    """Test ONNX inference"""
    onnx_path = "/home/qdrive/facial_recognition/modules/edge-device_dev/models/yolov8_face/yolov8n-face.onnx"
    
    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Get model info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"Input: {inputs[0].name} {inputs[0].shape}")
    print(f"Output: {outputs[0].name} {outputs[0].shape}")
    
    # Capture frame
    print("Capturing test frame...")
    frame = capture_test_frame()
    if frame is None:
        print("Using saved frame")
        frame = cv2.imread("/tmp/test_frame_pytorch.jpg")
    
    print(f"Frame shape: {frame.shape}")
    
    # Preprocess
    input_tensor, scale, padding = preprocess(frame)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    start = time.time()
    outputs = session.run(None, {inputs[0].name: input_tensor})
    elapsed = time.time() - start
    print(f"Inference time: {elapsed*1000:.1f}ms")
    
    output = outputs[0]
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    # Analyze output
    predictions = output[0].T  # Transpose to (N, features)
    print(f"Predictions shape after transpose: {predictions.shape}")
    
    # Check if format is [cx, cy, w, h, conf, ...] or something else
    print(f"\nFirst prediction (raw):")
    print(f"  {predictions[0, :10]}")
    
    # Try different confidence interpretations
    if predictions.shape[1] == 20:
        # Format: [cx, cy, w, h, conf, 5*kp]
        confidences = predictions[:, 4]
    elif predictions.shape[1] == 5:
        # Format: [cx, cy, w, h, conf]
        confidences = predictions[:, 4]
    else:
        # Maybe different format
        print(f"Unknown format with {predictions.shape[1]} features")
        confidences = predictions[:, 4]  # Try anyway
    
    print(f"\nConfidence analysis:")
    print(f"  Min: {confidences.min():.6f}")
    print(f"  Max: {confidences.max():.6f}")
    print(f"  Mean: {confidences.mean():.6f}")
    print(f"  >0.1: {(confidences > 0.1).sum()}")
    print(f"  >0.25: {(confidences > 0.25).sum()}")
    print(f"  >0.5: {(confidences > 0.5).sum()}")
    
    # Top predictions
    top_indices = np.argsort(confidences)[-5:][::-1]
    print(f"\nTop 5 predictions:")
    for idx in top_indices:
        print(f"  idx={idx}, conf={confidences[idx]:.4f}")
        print(f"    raw: {predictions[idx, :8]}")

if __name__ == "__main__":
    test_onnx()
