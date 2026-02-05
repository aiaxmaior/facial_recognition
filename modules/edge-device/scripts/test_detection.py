#!/usr/bin/env python3
"""
Test Face Detection - Verify face detection on test image

Usage:
    python scripts/test_detection.py
    python scripts/test_detection.py --backend yunet
    python scripts/test_detection.py --backend opencv
    python scripts/test_detection.py --backend retinaface
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def test_detector(image_path: str, backend: str):
    """Test a face detector on an image."""
    from deepface import DeepFace
    import cv2
    
    logger.info(f"Testing detector: {backend}")
    logger.info(f"Image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image")
        return None
    
    logger.info(f"Image shape: {img.shape}")
    
    # Time the detection
    start = time.time()
    
    try:
        faces = DeepFace.extract_faces(
            image_path,
            detector_backend=backend,
            enforce_detection=False,
            align=True
        )
        
        elapsed = time.time() - start
        
        logger.info(f"Detection time: {elapsed:.3f}s")
        logger.info(f"Faces found: {len(faces)}")
        
        for i, face in enumerate(faces):
            conf = face.get("confidence", "N/A")
            area = face.get("facial_area", {})
            face_img = face.get("face")
            
            logger.info(f"  Face {i}:")
            logger.info(f"    Confidence: {conf}")
            logger.info(f"    Area: x={area.get('x')}, y={area.get('y')}, w={area.get('w')}, h={area.get('h')}")
            if face_img is not None:
                logger.info(f"    Face image shape: {face_img.shape}")
        
        return faces
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Detection failed after {elapsed:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test face detection")
    parser.add_argument("--image", default="test_images/face.jpg", help="Path to test image")
    parser.add_argument("--backend", default=None, help="Specific backend to test (default: test all)")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    image_path = str(base_dir / args.image)
    
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return 1
    
    # Backends to test
    if args.backend:
        backends = [args.backend]
    else:
        backends = ["yunet", "opencv", "ssd", "retinaface"]
    
    results = {}
    
    print("\n" + "=" * 60)
    print("  Face Detection Test")
    print("=" * 60 + "\n")
    
    for backend in backends:
        print(f"\n--- Testing {backend} ---\n")
        faces = test_detector(image_path, backend)
        
        if faces:
            # Count faces with confidence > 0.5
            valid = [f for f in faces if f.get("confidence", 0) > 0.5]
            results[backend] = {
                "total": len(faces),
                "valid": len(valid),
                "best_conf": max(f.get("confidence", 0) for f in faces) if faces else 0
            }
        else:
            results[backend] = {"total": 0, "valid": 0, "best_conf": 0}
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    for backend, result in results.items():
        status = "OK" if result["valid"] > 0 else "FAIL"
        print(f"  {backend:12} | faces={result['total']:2} | valid={result['valid']:2} | best_conf={result['best_conf']:.3f} | {status}")
    
    print("=" * 60 + "\n")
    
    # Recommend best backend
    best = max(results.items(), key=lambda x: (x[1]["valid"], x[1]["best_conf"]))
    if best[1]["valid"] > 0:
        print(f"Recommended backend: {best[0]}")
        return 0
    else:
        print("WARNING: No backend detected faces with confidence > 0.5")
        return 1


if __name__ == "__main__":
    sys.exit(main())
