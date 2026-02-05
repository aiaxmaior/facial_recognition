#!/usr/bin/env python3
"""
Enrollment Migration Script

Migrates enrolled users from old Facenet512 format to new ArcFace format
by re-processing original enrollment images.

Source: ~/facial_recognition_old/enrolled_faces/*_debug/frame_*.jpg
Target: ~/facial_recognition/modules/edge-device/data/enrollments.db

Usage:
    python migrate_enrollments.py                    # Migrate all users
    python migrate_enrollments.py --dry-run          # Preview without changes
    python migrate_enrollments.py --user "John_Doe"  # Migrate specific user
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Source and target paths
OLD_ENROLLMENTS_DIR = Path.home() / "facial_recognition_old" / "enrolled_faces"
TARGET_DB_PATH = PROJECT_DIR / "data" / "enrollments.db"

# Model configuration (must match edge-device config)
MODEL_NAME = "ArcFace"
# Note: yolov8 not supported in current DeepFace version
# Available: opencv, ssd, mtcnn, retinaface, yunet
DETECTOR_BACKEND = "retinaface"


def find_enrollment_images(base_dir: Path) -> dict:
    """
    Find all enrollment images organized by user.
    
    Returns:
        Dict mapping user_name -> list of image paths
    """
    users = {}
    
    if not base_dir.exists():
        logger.error(f"Source directory not found: {base_dir}")
        return users
    
    # Find all *_debug directories
    for debug_dir in base_dir.glob("*_debug"):
        if not debug_dir.is_dir():
            continue
        
        # Extract user name (remove _debug suffix)
        user_name = debug_dir.name.replace("_debug", "")
        
        # Find all frame images
        images = sorted(debug_dir.glob("frame_*.jpg"))
        if images:
            users[user_name] = images
            logger.debug(f"Found {len(images)} images for {user_name}")
    
    return users


def extract_embedding(image_path: Path, model_name: str, detector: str) -> Optional[np.ndarray]:
    """
    Extract face embedding from an image using DeepFace.
    
    Returns:
        Numpy array of embedding, or None if extraction failed
    """
    try:
        from deepface import DeepFace
        import cv2
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return None
        
        # Extract embedding
        result = DeepFace.represent(
            img,
            model_name=model_name,
            detector_backend=detector,
            enforce_detection=False,
            align=True
        )
        
        if result and len(result) > 0:
            return np.array(result[0]["embedding"], dtype=np.float32)
        
        return None
        
    except Exception as e:
        logger.warning(f"Embedding extraction failed for {image_path}: {e}")
        return None


def compute_average_embedding(
    image_paths: List[Path],
    model_name: str,
    detector: str
) -> Tuple[Optional[np.ndarray], int]:
    """
    Compute average embedding from multiple images.
    
    Returns:
        (average_embedding, num_successful_extractions)
    """
    embeddings = []
    
    for img_path in image_paths:
        emb = extract_embedding(img_path, model_name, detector)
        if emb is not None:
            embeddings.append(emb)
    
    if not embeddings:
        return None, 0
    
    # Average all embeddings
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    
    # Normalize for cosine similarity
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    return avg_embedding, len(embeddings)


def migrate_user(
    user_name: str,
    image_paths: List[Path],
    db_manager,
    model_name: str,
    detector: str,
    dry_run: bool = False
) -> bool:
    """
    Migrate a single user's enrollment.
    
    Returns:
        True if successful
    """
    logger.info(f"Processing {user_name} ({len(image_paths)} images)...")
    
    # Compute average embedding
    embedding, num_images = compute_average_embedding(image_paths, model_name, detector)
    
    if embedding is None:
        logger.error(f"  FAILED: No valid embeddings extracted for {user_name}")
        return False
    
    logger.info(f"  Computed embedding from {num_images}/{len(image_paths)} images")
    logger.info(f"  Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would insert {user_name} into database")
        return True
    
    # Insert into database
    # Use user_name as both user_id and display_name for migrated users
    # In production, user_id would come from the central WFM system
    user_id = user_name.lower().replace(" ", "_").replace("-", "_")
    
    try:
        success = db_manager.upsert_enrollment(
            user_id=user_id,
            embedding=embedding,
            model=model_name,
            detector=detector,
            display_name=user_name,
            sync_version=0,
        )
        
        if success:
            logger.info(f"  SUCCESS: {user_name} enrolled as {user_id}")
            return True
        else:
            logger.error(f"  FAILED: Database insert failed for {user_name}")
            return False
            
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate enrollments from Facenet512 to ArcFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python migrate_enrollments.py                    # Migrate all
    python migrate_enrollments.py --dry-run          # Preview
    python migrate_enrollments.py --user Jack_Kelly  # Single user
    python migrate_enrollments.py --source /path/to/images
        """
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default=str(OLD_ENROLLMENTS_DIR),
        help=f"Source directory with *_debug folders (default: {OLD_ENROLLMENTS_DIR})"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=str(TARGET_DB_PATH),
        help=f"Target database path (default: {TARGET_DB_PATH})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Embedding model (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=DETECTOR_BACKEND,
        help=f"Face detector (default: {DETECTOR_BACKEND})"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Migrate only this user (by name)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to database"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enroll even if user already exists"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    source_dir = Path(args.source)
    target_db = Path(args.target)
    
    print("\n" + "=" * 60)
    print("  Enrollment Migration: Facenet512 → ArcFace")
    print("=" * 60)
    print(f"  Source:   {source_dir}")
    print(f"  Target:   {target_db}")
    print(f"  Model:    {args.model}")
    print(f"  Detector: {args.detector}")
    print(f"  Dry Run:  {args.dry_run}")
    print("=" * 60 + "\n")
    
    # Find all users with enrollment images
    users = find_enrollment_images(source_dir)
    
    if not users:
        logger.error("No enrollment images found!")
        sys.exit(1)
    
    logger.info(f"Found {len(users)} users with enrollment images")
    
    # Filter to specific user if requested
    if args.user:
        if args.user in users:
            users = {args.user: users[args.user]}
        else:
            logger.error(f"User '{args.user}' not found. Available: {list(users.keys())}")
            sys.exit(1)
    
    # Initialize database manager
    if not args.dry_run:
        try:
            from iot_integration import DatabaseManager
            db_manager = DatabaseManager(str(target_db))
            db_manager.initialize()
            logger.info(f"Database initialized: {target_db}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            sys.exit(1)
    else:
        db_manager = None
    
    # Load DeepFace models (once, for efficiency)
    logger.info("Loading face recognition models (this may take a moment)...")
    try:
        from deepface import DeepFace
        # Warm up the model
        DeepFace.build_model(args.model)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)
    
    # Migrate each user
    results = {"success": [], "failed": []}
    
    for user_name, image_paths in sorted(users.items()):
        success = migrate_user(
            user_name=user_name,
            image_paths=image_paths,
            db_manager=db_manager,
            model_name=args.model,
            detector=args.detector,
            dry_run=args.dry_run,
        )
        
        if success:
            results["success"].append(user_name)
        else:
            results["failed"].append(user_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("  Migration Summary")
    print("=" * 60)
    print(f"  Total users:    {len(users)}")
    print(f"  Successful:     {len(results['success'])}")
    print(f"  Failed:         {len(results['failed'])}")
    
    if results["failed"]:
        print(f"\n  Failed users: {results['failed']}")
    
    if args.dry_run:
        print("\n  [DRY RUN] No changes were made to the database")
    
    print("=" * 60 + "\n")
    
    # Exit code
    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
