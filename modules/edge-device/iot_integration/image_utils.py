"""
Image and video utilities for IoT event payloads.

Provides image/video compression and encoding for transmitting
event data to the IoT broker. All media is base64 encoded for
JSON payload compatibility.
"""

import base64
import io
import logging
import os
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compress_image_for_event(
    image: np.ndarray,
    target_size_kb: int = 50,
    initial_quality: int = 65,
    min_quality: int = 20,
    max_dimension: int = 320,
    crop_to_face: bool = True,
    face_bbox: Tuple[int, int, int, int] = None,
    padding_ratio: float = 0.3,
) -> Optional[str]:
    """
    Compress an image for IoT event transmission.
    
    Produces a base64-encoded JPEG that fits within the target size.
    
    Args:
        image: Input image (BGR or RGB format)
        target_size_kb: Target maximum size in kilobytes
        initial_quality: Starting JPEG quality (1-100)
        min_quality: Minimum acceptable quality
        max_dimension: Maximum width/height for output
        crop_to_face: If True and face_bbox provided, crop to face region
        face_bbox: Face bounding box [x, y, w, h]
        padding_ratio: Padding around face when cropping (as ratio of face size)
        
    Returns:
        Base64-encoded JPEG string, or None on failure
    """
    if image is None or image.size == 0:
        return None
    
    try:
        # Make a copy to avoid modifying original
        img = image.copy()
        
        # Crop to face region if requested
        if crop_to_face and face_bbox is not None:
            img = _crop_to_face(img, face_bbox, padding_ratio)
        
        # Resize if needed
        img = _resize_image(img, max_dimension)
        
        # Convert RGB to BGR if needed (for cv2.imencode)
        # Check if image appears to be RGB (OpenCV uses BGR)
        # This is a heuristic - assumes face images are roughly skin-toned
        
        # Compress with iterative quality reduction
        target_bytes = target_size_kb * 1024
        quality = initial_quality
        
        while quality >= min_quality:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, encoded = cv2.imencode(".jpg", img, encode_params)
            
            if not success:
                logger.error("Failed to encode image")
                return None
            
            size = len(encoded)
            
            if size <= target_bytes:
                # Success - encode to base64
                b64_string = base64.b64encode(encoded.tobytes()).decode("utf-8")
                logger.debug(
                    f"Image compressed: {size/1024:.1f}KB at quality={quality}"
                )
                return b64_string
            
            # Reduce quality and try again
            quality -= 10
        
        # Even at min quality, couldn't meet target
        # Return anyway with warning
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, min_quality]
        success, encoded = cv2.imencode(".jpg", img, encode_params)
        
        if success:
            logger.warning(
                f"Image exceeds target size: {len(encoded)/1024:.1f}KB "
                f"(target: {target_size_kb}KB)"
            )
            return base64.b64encode(encoded.tobytes()).decode("utf-8")
        
        return None
        
    except Exception as e:
        logger.error(f"Image compression failed: {e}")
        return None


def _crop_to_face(
    image: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    padding_ratio: float = 0.3,
) -> np.ndarray:
    """
    Crop image to face region with padding.
    
    Args:
        image: Input image
        face_bbox: Face bounding box [x, y, w, h]
        padding_ratio: Padding around face as ratio of face size
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x, y, fw, fh = face_bbox
    
    # Calculate padding
    pad_x = int(fw * padding_ratio)
    pad_y = int(fh * padding_ratio)
    
    # Calculate crop bounds with padding
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)
    
    # Crop
    return image[y1:y2, x1:x2]


def _resize_image(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """
    Resize image so largest dimension is at most max_dimension.
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
        
    Returns:
        Resized image (or original if already small enough)
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_dimension:
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    if w > h:
        new_w = max_dimension
        new_h = int(h * max_dimension / w)
    else:
        new_h = max_dimension
        new_w = int(w * max_dimension / h)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_event_image(b64_string: str) -> Optional[np.ndarray]:
    """
    Decode a base64 event image back to numpy array.
    
    Args:
        b64_string: Base64-encoded JPEG string
        
    Returns:
        Decoded image as numpy array (BGR format), or None on failure
    """
    try:
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def encode_embedding_b64(embedding: np.ndarray) -> str:
    """
    Encode a face embedding to base64 string.
    
    Args:
        embedding: Face embedding as numpy array (float32)
        
    Returns:
        Base64-encoded string
    """
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode("utf-8")


def decode_embedding_b64(b64_string: str, dim: int = 512) -> np.ndarray:
    """
    Decode a base64 face embedding back to numpy array.
    
    Args:
        b64_string: Base64-encoded embedding string
        dim: Expected embedding dimension (for validation)
        
    Returns:
        Embedding as numpy float32 array
    """
    embedding_bytes = base64.b64decode(b64_string)
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
    
    if len(embedding) != dim:
        logger.warning(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {dim}"
        )
    
    return embedding


def extract_face_thumbnail(
    image: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    size: int = 128,
) -> Optional[np.ndarray]:
    """
    Extract a square face thumbnail from image.
    
    Args:
        image: Input image
        face_bbox: Face bounding box [x, y, w, h]
        size: Output thumbnail size (square)
        
    Returns:
        Square thumbnail image, or None on failure
    """
    try:
        # Crop to face with some padding
        face_crop = _crop_to_face(image, face_bbox, padding_ratio=0.2)
        
        # Make square (center crop if needed)
        h, w = face_crop.shape[:2]
        if w != h:
            min_dim = min(w, h)
            x_start = (w - min_dim) // 2
            y_start = (h - min_dim) // 2
            face_crop = face_crop[y_start:y_start+min_dim, x_start:x_start+min_dim]
        
        # Resize to target size
        thumbnail = cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)
        
        return thumbnail
        
    except Exception as e:
        logger.error(f"Failed to extract face thumbnail: {e}")
        return None


# =============================================================================
# Video Clip Utilities
# =============================================================================

def encode_video_clip_b64(file_path: str) -> Optional[str]:
    """
    Read a video clip file and encode to base64 string.
    
    The base64 string can be included in JSON payloads for transmission
    to the IoT broker. The receiving end decodes back to binary MP4.
    
    Args:
        file_path: Path to MP4 video file
        
    Returns:
        Base64-encoded string of the video file, or None on failure
    """
    try:
        with open(file_path, "rb") as f:
            video_bytes = f.read()
        
        b64_string = base64.b64encode(video_bytes).decode("utf-8")
        
        logger.debug(
            f"Video encoded: {len(video_bytes)/1024:.1f}KB -> "
            f"{len(b64_string)/1024:.1f}KB base64"
        )
        
        return b64_string
        
    except FileNotFoundError:
        logger.error(f"Video file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to encode video clip: {e}")
        return None


def decode_video_clip_b64(b64_string: str, output_path: str) -> bool:
    """
    Decode a base64 video string and save to file.
    
    Args:
        b64_string: Base64-encoded video data
        output_path: Path to save the decoded MP4 file
        
    Returns:
        True if successful
    """
    try:
        video_bytes = base64.b64decode(b64_string)
        
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        
        logger.debug(f"Video decoded: {len(video_bytes)/1024:.1f}KB -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to decode video clip: {e}")
        return False


def get_video_clip_size(file_path: str) -> Optional[int]:
    """
    Get the size of a video clip file in bytes.
    
    Args:
        file_path: Path to video file
        
    Returns:
        File size in bytes, or None if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except Exception:
        return None
