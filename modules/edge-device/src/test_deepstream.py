#!/usr/bin/env python3
"""
Test script for DeepStream + TensorRT hybrid pipeline components.
Tests:
1. GStreamer RTSP decode with hardware acceleration (nvv4l2decoder)
2. TensorRT YOLOv8-face inference
"""

import sys
import time
import json
import numpy as np

# Test 1: GStreamer/DeepStream video decode
def test_gstreamer_decode():
    """Test hardware-accelerated RTSP decode using GStreamer."""
    print("\n" + "="*60)
    print("TEST 1: GStreamer RTSP Decode (Hardware Accelerated)")
    print("="*60)
    
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib
        Gst.init(None)
        print("[OK] GStreamer initialized")
    except Exception as e:
        print(f"[FAIL] GStreamer import failed: {e}")
        return False, None
    
    # Load config
    config_path = "/home/qdrive/facial_recognition/modules/edge-device/config/config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    rtsp_url = config["camera"]["rtsp_url"]
    print(f"[INFO] RTSP URL: {rtsp_url}")
    
    # Build pipeline: RTSP -> HW decode -> convert -> appsink
    pipeline_str = f'''
        rtspsrc location="{rtsp_url}" latency=200 !
        rtph264depay !
        h264parse !
        nvv4l2decoder !
        nvvideoconvert !
        video/x-raw,format=BGRx,width=640,height=360 !
        appsink name=sink emit-signals=false max-buffers=2 drop=true sync=false
    '''
    
    print(f"[INFO] Building pipeline...")
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
        print("[OK] Pipeline created")
    except Exception as e:
        print(f"[FAIL] Pipeline creation failed: {e}")
        return False, None
    
    # Get appsink
    appsink = pipeline.get_by_name("sink")
    
    # Start pipeline
    print("[INFO] Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[FAIL] Failed to start pipeline")
        return False, None
    
    # Wait for pipeline to be ready
    print("[INFO] Waiting for stream to stabilize (3s)...")
    time.sleep(3)
    
    # Pull frames directly using blocking pull
    print("[INFO] Pulling frames (5 seconds)...")
    frames_received = []
    start_time = time.time()
    timeout = 5  # seconds
    
    while time.time() - start_time < timeout:
        sample = appsink.emit("try-pull-sample", int(0.1 * Gst.SECOND))
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)
            width = struct.get_value("width")
            height = struct.get_value("height")
            
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                # BGRx format: 4 bytes per pixel
                frame = np.ndarray(
                    shape=(height, width, 4),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                # Convert BGRx to BGR
                frame_bgr = frame[:, :, :3].copy()
                frames_received.append(frame_bgr)
                buf.unmap(map_info)
    
    # Stop pipeline
    pipeline.set_state(Gst.State.NULL)
    
    elapsed = time.time() - start_time
    fps = len(frames_received) / elapsed if elapsed > 0 else 0
    
    print(f"[INFO] Captured {len(frames_received)} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
    
    if len(frames_received) > 0:
        frame = frames_received[-1]
        print(f"[OK] Frame shape: {frame.shape}, dtype: {frame.dtype}")
        print("[OK] GStreamer decode test PASSED")
        return True, frame
    else:
        print("[FAIL] No frames received")
        return False, None


# Test 2: TensorRT inference
def test_tensorrt_inference(test_frame=None):
    """Test TensorRT YOLOv8-face inference."""
    print("\n" + "="*60)
    print("TEST 2: TensorRT YOLOv8-face Inference")
    print("="*60)
    
    engine_path = "/home/qdrive/facial_recognition/modules/edge-device/models/yolov8_face/yolov8n-face.engine"
    
    try:
        import tensorrt as trt
        print(f"[OK] TensorRT version: {trt.__version__}")
    except ImportError:
        print("[FAIL] TensorRT not available")
        return False
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("[OK] PyCUDA initialized")
    except ImportError:
        print("[FAIL] PyCUDA not available - trying without explicit init")
    
    # Load engine
    print(f"[INFO] Loading engine: {engine_path}")
    
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        print(f"[OK] Engine loaded")
    except Exception as e:
        print(f"[FAIL] Engine load failed: {e}")
        return False
    
    # Print engine info
    print(f"[INFO] Engine bindings:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"       [{i}] {name}: {shape} ({dtype}) - {mode}")
    
    # Create execution context
    try:
        context = engine.create_execution_context()
        print("[OK] Execution context created")
    except Exception as e:
        print(f"[FAIL] Context creation failed: {e}")
        return False
    
    # Prepare input
    input_shape = (1, 3, 640, 640)  # NCHW
    
    if test_frame is not None:
        print("[INFO] Using frame from GStreamer test")
        # Preprocess frame
        import cv2
        frame_resized = cv2.resize(test_frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_data = frame_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.ascontiguousarray(np.expand_dims(input_data, 0).astype(np.float32))
    else:
        print("[INFO] Using random test input")
        input_data = np.ascontiguousarray(np.random.rand(*input_shape).astype(np.float32))
    
    print(f"[INFO] Input shape: {input_data.shape}")
    
    # Allocate buffers
    try:
        import pycuda.driver as cuda
        
        # Get tensor info
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        output_shape = engine.get_tensor_shape(output_name)
        
        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        output_size = int(np.prod(output_shape)) * 4  # float32
        d_output = cuda.mem_alloc(output_size)
        
        # Copy input to device
        cuda.memcpy_htod(d_input, input_data)
        
        # Set tensor addresses
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        
        # Run inference
        print("[INFO] Running inference...")
        stream = cuda.Stream()
        
        start = time.time()
        for _ in range(10):  # Warm up + timing
            context.execute_async_v3(stream.handle)
        stream.synchronize()
        elapsed = (time.time() - start) / 10
        
        print(f"[OK] Inference time: {elapsed*1000:.2f}ms ({1/elapsed:.1f} FPS)")
        
        # Copy output back
        output_data = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_data, d_output)
        
        print(f"[OK] Output shape: {output_data.shape}")
        print(f"[INFO] Output sample (first 5 detections, first 6 values):")
        print(output_data[0, :5, :6])
        
        # Basic detection parsing (YOLOv8-pose format)
        # Output: [batch, num_detections, 4 + 1 + 5*3] = [1, 8400, 20]
        # Format: [x, y, w, h, conf, kp1_x, kp1_y, kp1_conf, ...]
        detections = output_data[0]  # [8400, 20]
        
        # Filter by confidence (column 4)
        conf_threshold = 0.5
        mask = detections[:, 4] > conf_threshold
        valid_detections = detections[mask]
        
        print(f"[INFO] Detections above {conf_threshold} confidence: {len(valid_detections)}")
        
        if len(valid_detections) > 0:
            print("[INFO] Top detection:")
            top = valid_detections[0]
            print(f"       Box (xywh): [{top[0]:.1f}, {top[1]:.1f}, {top[2]:.1f}, {top[3]:.1f}]")
            print(f"       Confidence: {top[4]:.3f}")
        
        print("[OK] TensorRT inference test PASSED")
        return True
        
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("DeepStream + TensorRT Hybrid Pipeline Test")
    print("="*60)
    
    # Test 1: GStreamer decode
    gst_ok, test_frame = test_gstreamer_decode()
    
    # Test 2: TensorRT inference
    trt_ok = test_tensorrt_inference(test_frame)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"GStreamer RTSP Decode: {'PASS' if gst_ok else 'FAIL'}")
    print(f"TensorRT Inference:    {'PASS' if trt_ok else 'FAIL'}")
    print("="*60)
    
    if gst_ok and trt_ok:
        print("\nAll tests passed! Ready to build hybrid pipeline.")
        return 0
    else:
        print("\nSome tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
