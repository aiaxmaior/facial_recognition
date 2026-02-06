#!/usr/bin/env python3
"""
Simplified TRT inference test with detailed error checking.
"""
import sys
for syspath in ['/usr/lib/python3.10/dist-packages', '/usr/lib/python3/dist-packages']:
    if syspath not in sys.path:
        sys.path.append(syspath)

import cv2
import numpy as np
import time

def test_trt():
    import tensorrt as trt
    import pycuda.driver as cuda
    
    # Initialize CUDA
    cuda.init()
    print(f"CUDA devices: {cuda.Device.count()}")
    
    device = cuda.Device(0)
    print(f"Using device: {device.name()}")
    
    ctx = device.make_context()
    print("CUDA context created")
    
    try:
        engine_path = "/home/qdrive/facial_recognition/modules/edge-device/models/yolov8_face/yolov8n-face.engine"
        
        logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"Loading engine: {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("ERROR: Failed to deserialize engine")
            return
        
        print("Engine loaded successfully")
        
        context = engine.create_execution_context()
        if context is None:
            print("ERROR: Failed to create execution context")
            return
        
        print("Execution context created")
        
        # Get tensor info
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        input_shape = list(engine.get_tensor_shape(input_name))
        output_shape = list(engine.get_tensor_shape(output_name))
        
        print(f"Input: {input_name} {input_shape}")
        print(f"Output: {output_name} {output_shape}")
        
        # Allocate memory
        input_size = int(np.prod(input_shape) * 4)
        output_size = int(np.prod(output_shape) * 4)
        print(f"Allocating {input_size} bytes for input, {output_size} bytes for output")
        
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        stream = cuda.Stream()
        print("GPU memory allocated")
        
        # Set tensor addresses
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        print("Tensor addresses set")
        
        # Prepare test input (same image as ONNX test)
        print("\nLoading test image...")
        frame = cv2.imread("/tmp/test_frame_pytorch.jpg")
        if frame is None:
            print("Test image not found, capturing from RTSP...")
            rtsp_url = "rtsp://admin:Fanatec2025@192.168.13.119/Preview_01_sub"
            cap = cv2.VideoCapture(rtsp_url)
            for _ in range(10):
                ret, frame = cap.read()
            cap.release()
        
        print(f"Frame shape: {frame.shape}")
        
        # Preprocess (same as ONNX test)
        h, w = frame.shape[:2]
        input_size_px = 640
        scale = min(input_size_px / w, input_size_px / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((input_size_px, input_size_px, 3), 114, dtype=np.uint8)
        pad_x = (input_size_px - new_w) // 2
        pad_y = (input_size_px - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = normalized.transpose(2, 0, 1)
        input_tensor = np.ascontiguousarray(chw[np.newaxis, ...])
        
        print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"Input tensor range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
        print(f"Input is contiguous: {input_tensor.flags['C_CONTIGUOUS']}")
        
        # Copy to GPU
        print("\nCopying input to GPU...")
        cuda.memcpy_htod(d_input, input_tensor)
        print("Input copied")
        
        # Run inference (synchronous version first)
        print("Running inference (sync)...")
        output_buffer = np.empty(output_shape, dtype=np.float32)
        output_buffer = np.ascontiguousarray(output_buffer)
        
        success = context.execute_v2([int(d_input), int(d_output)])
        print(f"execute_v2 returned: {success}")
        
        if not success:
            print("ERROR: execute_v2 failed!")
            # Try async version
            print("Trying execute_async_v3...")
            success2 = context.execute_async_v3(stream.handle)
            stream.synchronize()
            print(f"execute_async_v3 returned: {success2}")
        
        # Copy output back
        print("Copying output from GPU...")
        cuda.memcpy_dtoh(output_buffer, d_output)
        print("Output copied")
        
        print(f"\nOutput shape: {output_buffer.shape}")
        print(f"Output range: [{output_buffer.min():.6f}, {output_buffer.max():.6f}]")
        print(f"Output mean: {output_buffer.mean():.6f}")
        print(f"Output sum: {output_buffer.sum():.6f}")
        
        # Check if output is all zeros
        if np.allclose(output_buffer, 0):
            print("\nWARNING: Output is all zeros!")
        
        # Analyze output
        predictions = output_buffer[0].T
        confidences = predictions[:, 4]
        
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
        
        d_input.free()
        d_output.free()
        print("\nGPU memory freed")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ctx.pop()
        ctx.detach()
        print("CUDA context cleaned up")

if __name__ == "__main__":
    test_trt()
