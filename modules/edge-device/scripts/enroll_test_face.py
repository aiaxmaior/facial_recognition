#!/usr/bin/env python3
"""
Enroll Test Face - Create a test enrollment from test_images/face.jpg

Usage:
    python scripts/enroll_test_face.py
    python scripts/enroll_test_face.py --user-id "EMP-001" --name "Test User"
    python scripts/enroll_test_face.py --image path/to/face.jpg
"""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Enroll a test face")
    parser.add_argument("--user-id", default="TEST-001", help="User ID for enrollment")
    parser.add_argument("--name", default="Test User", help="Display name")
    parser.add_argument("--image", default="test_images/face.jpg", help="Path to face image")
    parser.add_argument("--db", default="data/enrollments.db", help="Path to database")
    parser.add_argument("--model", default="ArcFace", help="Recognition model")
    parser.add_argument("--detector", default="yunet", help="Face detector backend")
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    image_path = base_dir / args.image
    db_path = base_dir / args.db
    
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1
    
    logger.info(f"Loading image: {image_path}")
    logger.info(f"Database: {db_path}")
    
    try:
        from deepface import DeepFace
        import cv2
        
        # Load and verify image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return 1
        
        logger.info(f"Image shape: {img.shape}")
        
        # Extract face and embedding
        logger.info(f"Extracting embedding with {args.model} model and {args.detector} detector...")
        
        # First detect face
        faces = DeepFace.extract_faces(
            str(image_path),
            detector_backend=args.detector,
            enforce_detection=True,
            align=True
        )
        
        if not faces:
            logger.error("No face detected in image!")
            return 1
        
        logger.info(f"Detected {len(faces)} face(s)")
        face = faces[0]
        logger.info(f"Face confidence: {face.get('confidence', 'N/A')}")
        logger.info(f"Face area: {face.get('facial_area', {})}")
        
        # Get embedding
        result = DeepFace.represent(
            str(image_path),
            model_name=args.model,
            detector_backend=args.detector,
            enforce_detection=True
        )
        
        if not result:
            logger.error("Failed to extract embedding!")
            return 1
        
        embedding = np.array(result[0]["embedding"], dtype=np.float32)
        logger.info(f"Embedding shape: {embedding.shape}")
        logger.info(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        # Save to database
        from iot_integration.db_manager import DatabaseManager
        
        db = DatabaseManager(str(db_path), dev_mode=True)
        db.initialize()
        
        success = db.upsert_enrollment(
            user_id=args.user_id,
            embedding=embedding,
            model=args.model,
            detector=args.detector,
            display_name=args.name,
            sync_version=1
        )
        
        if success:
            logger.info(f"Successfully enrolled: {args.user_id} ({args.name})")
            
            # Verify enrollment
            count = db.get_enrollment_count()
            logger.info(f"Total enrollments in database: {count}")
            
            # Verify we can retrieve it
            enrollments = db.get_all_enrollments()
            if args.user_id in enrollments:
                logger.info(f"Verified: {args.user_id} retrievable from database")
            else:
                logger.warning(f"Warning: {args.user_id} not found after insert")
            
            return 0
        else:
            logger.error("Failed to save enrollment to database")
            return 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
