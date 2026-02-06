#!/usr/bin/env python3
"""
Standalone test for TRT face detector to debug inference issues.
"""
import sys
# Append system paths for tensorrt, pycuda
for syspath in ['/usr/lib/python3.10/dist-packages', '/usr/lib/python3/dist-packages']:
    if syspath not in sys.path:
        sys.path.append(syspath)

import cv2
import numpy as np
import time

# Test with a static image - capture one from RTSP
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

def test_raw_inference():
    """Test TRT inference directly without full pipeline"""
    import tensorrt as trt
    import pycuda.driver as cuda
    
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    
    try:
        engine_path = "/home/qdrive/facial_recognition/modules/edge-device/models/yolov8_face/yolov8n-face.engine"
        
        logger = trt.Logger(trt.Logger.VERBOSE)
        
        print(f"Loading engine: {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Get tensor info
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        input_shape = engine.get_tensor_shape(input_name)
        output_shape = engine.get_tensor_shape(output_name)
        
        print(f"Input: {input_name} {input_shape}")
        print(f"Output: {output_name} {output_shape}")
        
        # Allocate memory
        input_size = int(np.prod(input_shape) * 4)
        output_size = int(np.prod(output_shape) * 4)
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        stream = cuda.Stream()
        
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        
        # Capture test frame
        print("Capturing test frame from RTSP...")
        frame = capture_test_frame()
        if frame is None:
            # Use synthetic test pattern
            print("Using synthetic test pattern")
            frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        
        print(f"Frame shape: {frame.shape}")
        cv2.imwrite("/tmp/test_frame.jpg", frame)
        print("Saved test frame to /tmp/test_frame.jpg")
        
        # Preprocess
        h, w = frame.shape[:2]
        input_size_px = 640
        scale = min(input_size_px / w, input_size_px / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((input_size_px, input_size_px, 3), 114, dtype=np.uint8)
        pad_x = (input_size_px - new_w) // 2
        pad_y = (input_size_px - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # Convert to model input format
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = normalized.transpose(2, 0, 1)
        input_tensor = np.ascontiguousarray(chw[np.newaxis, ...])
        
        print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"Input tensor range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
        
        # Run inference
        print("Running inference...")
        cuda.memcpy_htod_async(d_input, input_tensor, stream)
        success = context.execute_async_v3(stream.handle)
        print(f"execute_async_v3 returned: {success}")
        
        output_buffer = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_buffer, d_output, stream)
        stream.synchronize()
        
        print(f"Output shape: {output_buffer.shape}")
        print(f"Output range: [{output_buffer.min():.6f}, {output_buffer.max():.6f}]")
        print(f"Output mean: {output_buffer.mean():.6f}")
        
        # Analyze output in detail
        predictions = output_buffer[0].T  # (8400, 20)
        
        # Format: [cx, cy, w, h, conf, kp1_x, kp1_y, kp1_vis, ..., kp5_x, kp5_y, kp5_vis]
        boxes = predictions[:, :4]
        confidences = predictions[:, 4]
        
        print(f"\nConfidence analysis:")
        print(f"  Min: {confidences.min():.6f}")
        print(f"  Max: {confidences.max():.6f}")
        print(f"  Mean: {confidences.mean():.6f}")
        print(f"  Std: {confidences.std():.6f}")
        print(f"  >0.01: {(confidences > 0.01).sum()}")
        print(f"  >0.05: {(confidences > 0.05).sum()}")
        print(f"  >0.1: {(confidences > 0.1).sum()}")
        print(f"  >0.25: {(confidences > 0.25).sum()}")
        print(f"  >0.5: {(confidences > 0.5).sum()}")
        
        # Show top predictions
        top_indices = np.argsort(confidences)[-10:][::-1]
        print(f"\nTop 10 predictions:")
        for idx in top_indices:
            print(f"  conf={confidences[idx]:.6f}, box={boxes[idx]}")
        
        # Also check different interpretations of the output
        print(f"\nAlternative output interpretations:")
        
        # Maybe output is (1, 8400, 20)?
        alt1 = output_buffer[0]
        print(f"  As (1, 8400, 20) - row 0: {alt1[0, :5]}")
        print(f"  As (1, 8400, 20) - row -1: {alt1[-1, :5]}")
        
        # Check raw output values at different positions
        print(f"\nRaw output samples:")
        print(f"  output_buffer[0, 0, :5]: {output_buffer[0, 0, :5]}")
        print(f"  output_buffer[0, 4, :5]: {output_buffer[0, 4, :5]}")  # Confidence channel
        print(f"  output_buffer[0, :, 0]: {output_buffer[0, :, 0]}")  # First detection
        
        d_input.free()
        d_output.free()
        
    finally:
        ctx.pop()
        ctx.detach()

if __name__ == "__main__":
    test_raw_inference()
