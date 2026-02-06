"""
TensorRT YOLOv8-face Detector

Runs YOLOv8n-face model using TensorRT for GPU-accelerated face detection.
Returns face bounding boxes and optional facial landmarks.

Model output format (YOLOv8-pose trained on faces):
    Shape: (1, 20, 8400) = [batch, features, num_predictions]
    Features: [cx, cy, w, h, conf, kp1_x, kp1_y, kp1_vis, ..., kp5_x, kp5_y, kp5_vis]
    
    Keypoints (5 facial landmarks):
        0: left_eye
        1: right_eye  
        2: nose
        3: left_mouth
        4: right_mouth
"""

import sys
# Add system site-packages for tensorrt when running in venv
if '/usr/lib/python3.10/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

import time
import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import tensorrt as trt
import pycuda.driver as cuda

# Don't use autoinit - we'll manage context manually to avoid conflicts with GStreamer
# Initialize CUDA driver
cuda.init()

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Single face detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[np.ndarray] = None  # (5, 2) array of [x, y] points
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3)
        }
        if self.landmarks is not None:
            result["landmarks"] = self.landmarks.tolist()
        return result


class TRTFaceDetector:
    """
    TensorRT-accelerated YOLOv8 face detector.
    
    Usage:
        detector = TRTFaceDetector(engine_path, conf_threshold=0.5)
        faces = detector.detect(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            confidence = face.confidence
    """
    
    def __init__(
        self,
        engine_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        max_detections: int = 100
    ):
        """
        Initialize TensorRT face detector.
        
        Args:
            engine_path: Path to TensorRT engine file
            input_size: Model input size (assumes square)
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections to return
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Create CUDA context for this detector
        self.cuda_device = cuda.Device(0)
        self.cuda_ctx = self.cuda_device.make_context()
        
        try:
            # Load TensorRT engine
            logger.info(f"Loading TensorRT engine: {engine_path}")
            self.logger = trt.Logger(trt.Logger.WARNING)
            
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # Get tensor info
            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)
            self.input_shape = self.engine.get_tensor_shape(self.input_name)
            self.output_shape = self.engine.get_tensor_shape(self.output_name)
            
            logger.info(f"Input: {self.input_name} {self.input_shape}")
            logger.info(f"Output: {self.output_name} {self.output_shape}")
            
            # Allocate GPU memory
            self.input_size_bytes = int(np.prod(self.input_shape) * 4)  # float32
            self.output_size_bytes = int(np.prod(self.output_shape) * 4)
            
            self.d_input = cuda.mem_alloc(self.input_size_bytes)
            self.d_output = cuda.mem_alloc(self.output_size_bytes)
            self.stream = cuda.Stream()
            
            # Pre-allocate output buffer
            self.output_buffer = np.empty(self.output_shape, dtype=np.float32)
            
            # Set tensor addresses
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            
            # Warm up (context is already active)
            self._warmup_internal()
            
            logger.info("TRT Face Detector initialized")
        finally:
            # Pop context so other CUDA operations can use GPU
            self.cuda_ctx.pop()
    
    def _warmup_internal(self, iterations: int = 3):
        """Warm up the engine (context must already be active)."""
        logger.debug("Warming up TensorRT engine...")
        dummy_input = np.zeros(self.input_shape, dtype=np.float32)
        dummy_input = np.ascontiguousarray(dummy_input)
        
        for _ in range(iterations):
            cuda.memcpy_htod_async(self.d_input, dummy_input, self.stream)
            self.context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for model input.
        
        Args:
            frame: BGR frame, any size
            
        Returns:
            input_tensor: (1, 3, H, W) float32 tensor
            scale: Scale factor applied
            padding: (pad_x, pad_y) padding applied
        """
        h, w = frame.shape[:2]
        
        # Calculate scale to fit in input_size while maintaining aspect ratio
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (letterbox)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert BGR to RGB, normalize, transpose to CHW
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        input_tensor = np.ascontiguousarray(chw[np.newaxis, ...])
        
        return input_tensor, scale, (pad_x, pad_y)
    
    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[FaceDetection]:
        """
        Post-process model output to get face detections.
        
        Args:
            output: Model output (1, 20, 8400)
            scale: Scale factor from preprocessing
            padding: (pad_x, pad_y) from preprocessing
            orig_shape: Original frame (height, width)
            
        Returns:
            List of FaceDetection objects
        """
        # Output shape: (1, 20, 8400) -> transpose to (8400, 20)
        predictions = output[0].T  # (8400, 20)
        
        # Extract components
        # Format: [cx, cy, w, h, conf, kp1_x, kp1_y, kp1_vis, ..., kp5_x, kp5_y, kp5_vis]
        boxes = predictions[:, :4]  # (8400, 4) - cx, cy, w, h
        confidences = predictions[:, 4]  # (8400,)
        keypoints = predictions[:, 5:].reshape(-1, 5, 3)  # (8400, 5, 3) - x, y, vis
        
        # Debug: log confidence statistics (first 100 frames only)
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1
        if self._debug_frame_count <= 10:
            logger.info(f"[DEBUG] Frame {self._debug_frame_count}: conf min={confidences.min():.4f}, max={confidences.max():.4f}, mean={confidences.mean():.4f}, >0.1: {(confidences > 0.1).sum()}, >0.25: {(confidences > 0.25).sum()}, >0.5: {(confidences > 0.5).sum()}")
        
        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        keypoints = keypoints[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from cx, cy, w, h to x1, y1, x2, y2
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Remove padding and scale back to original size
        pad_x, pad_y = padding
        orig_h, orig_w = orig_shape
        
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale
        
        # Scale keypoints
        keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / scale
        
        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
        
        # Apply NMS
        indices = self._nms(boxes_xyxy, confidences, self.nms_threshold)
        
        # Limit detections
        indices = indices[:self.max_detections]
        
        # Build results
        detections = []
        for idx in indices:
            bbox = tuple(boxes_xyxy[idx].astype(int))
            conf = float(confidences[idx])
            kps = keypoints[idx, :, :2]  # Just x, y (drop visibility)
            
            detections.append(FaceDetection(
                bbox=bbox,
                confidence=conf,
                landmarks=kps
            ))
        
        return detections
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> List[int]:
        """
        Non-maximum suppression.
        
        Args:
            boxes: (N, 4) array of boxes in xyxy format
            scores: (N,) array of confidence scores
            iou_threshold: IoU threshold
            
        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Compute IoU with rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in frame.
        
        Args:
            frame: BGR frame, any size
            
        Returns:
            List of FaceDetection objects
        """
        orig_shape = frame.shape[:2]
        
        # Preprocess
        input_tensor, scale, padding = self._preprocess(frame)
        
        # Push CUDA context for this thread
        self.cuda_ctx.push()
        
        try:
            # Copy to GPU
            cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)
            
            # Run inference
            self.context.execute_async_v3(self.stream.handle)
            
            # Copy result back
            cuda.memcpy_dtoh_async(self.output_buffer, self.d_output, self.stream)
            self.stream.synchronize()
        finally:
            # Pop CUDA context
            self.cuda_ctx.pop()
        
        # Postprocess
        detections = self._postprocess(self.output_buffer, scale, padding, orig_shape)
        
        return detections
    
    def detect_with_timing(self, frame: np.ndarray) -> Tuple[List[FaceDetection], float]:
        """
        Detect faces with timing info.
        
        Returns:
            (detections, inference_time_ms)
        """
        start = time.perf_counter()
        detections = self.detect(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return detections, elapsed_ms
    
    def __del__(self):
        """Cleanup GPU resources."""
        try:
            if hasattr(self, 'cuda_ctx'):
                self.cuda_ctx.push()
                if hasattr(self, 'd_input'):
                    self.d_input.free()
                if hasattr(self, 'd_output'):
                    self.d_output.free()
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
        except:
            pass


# Test
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    engine_path = "models/yolov8_face/yolov8n-face.engine"
    
    print("Initializing TRT Face Detector...")
    detector = TRTFaceDetector(
        engine_path,
        conf_threshold=0.5,
        nms_threshold=0.45
    )
    
    # Test with dummy image
    print("\nTesting with dummy image...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    faces, time_ms = detector.detect_with_timing(dummy_frame)
    print(f"Detected {len(faces)} faces in {time_ms:.2f}ms")
    
    # Benchmark
    print("\nBenchmarking (100 iterations)...")
    times = []
    for _ in range(100):
        _, t = detector.detect_with_timing(dummy_frame)
        times.append(t)
    
    print(f"Average: {np.mean(times):.2f}ms")
    print(f"Min: {np.min(times):.2f}ms")
    print(f"Max: {np.max(times):.2f}ms")
    print(f"FPS: {1000 / np.mean(times):.1f}")
