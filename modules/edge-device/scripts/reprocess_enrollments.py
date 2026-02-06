#!/usr/bin/env python3
"""
Reprocess enrollment images into the current matching schema.

Reads original test users' images from a directory (e.g. enrolled_users or
enrolled_faces), extracts embeddings using the same detector/model as the
recognition pipeline (yolov8n + ArcFace by default), and upserts into
enrollments.db so recognition matches correctly.

Usage:
    python scripts/reprocess_enrollments.py
    python scripts/reprocess_enrollments.py --source enrolled_users --db data/enrollments.db
    python scripts/reprocess_enrollments.py --source /path/to/enrolled_faces --config config/config.json
    python scripts/reprocess_enrollments.py --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project root (edge-device_dev)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SOURCE = PROJECT_DIR / "enrolled_users"
DEFAULT_DB = PROJECT_DIR / "data" / "enrollments.db"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "config.json"

# Must match recognition pipeline (config.json recognition section)
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "yolov8n"

# Image extensions to consider
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_recognition_config(config_path: Path) -> Tuple[str, str]:
    """Read model and detector_backend from config JSON. Returns (model, detector)."""
    if not config_path.exists():
        return MODEL_NAME, DETECTOR_BACKEND
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        rec = cfg.get("recognition", {})
        return (
            rec.get("model", MODEL_NAME),
            rec.get("detector_backend", DETECTOR_BACKEND),
        )
    except Exception as e:
        logger.warning(f"Could not read config {config_path}: {e}")
        return MODEL_NAME, DETECTOR_BACKEND


def find_enrollment_images(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Find enrollment images organized by user.

    Supports:
      1) *_debug/frame_*.jpg (same as migrate_enrollments)
      2) Any subdir with images: user_id/*.jpg etc.

    Returns:
        Dict mapping user_id or user_name -> list of image paths
    """
    users: Dict[str, List[Path]] = {}

    if not base_dir.exists():
        logger.error(f"Source directory not found: {base_dir}")
        return users

    # Layout 1: *_debug/frame_*.jpg
    for debug_dir in base_dir.glob("*_debug"):
        if not debug_dir.is_dir():
            continue
        name = debug_dir.name.replace("_debug", "")
        images = sorted(debug_dir.glob("frame_*.jpg"))
        if images:
            users[name] = images
            logger.debug(f"Found {len(images)} images for {name} (*_debug)")

    # Layout 2: subdirs with any images (e.g. enrolled_users/user_1/*.jpg)
    for subdir in base_dir.iterdir():
        if not subdir.is_dir() or subdir.name.endswith("_debug"):
            continue
        images = [
            p
            for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            images = sorted(images)
            # Use subdir name as user id (e.g. user_3, john_doe)
            users[subdir.name] = images
            logger.debug(f"Found {len(images)} images for {subdir.name}")

    return users


def extract_embedding(
    image_path: Path,
    model_name: str,
    detector: str,
) -> Optional[np.ndarray]:
    """Extract face embedding with current pipeline detector/model."""
    try:
        from deepface import DeepFace

        img = __import__("cv2").imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return None

        result = DeepFace.represent(
            img,
            model_name=model_name,
            detector_backend=detector,
            enforce_detection=False,
            align=True,
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
    detector: str,
) -> Tuple[Optional[np.ndarray], int]:
    """Average embeddings from multiple images; L2-normalize. Returns (embedding, num_ok)."""
    embeddings = []
    for p in image_paths:
        emb = extract_embedding(p, model_name, detector)
        if emb is not None:
            embeddings.append(emb)
    if not embeddings:
        return None, 0
    avg = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg, len(embeddings)


def reprocess_user(
    user_id: str,
    image_paths: List[Path],
    db_manager,
    model_name: str,
    detector: str,
    display_name: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Extract embedding for one user and upsert into DB."""
    logger.info(f"Processing {user_id} ({len(image_paths)} images)...")
    embedding, num_ok = compute_average_embedding(image_paths, model_name, detector)
    if embedding is None:
        logger.error(f"  FAILED: No valid embeddings for {user_id}")
        return False
    logger.info(f"  Embedding from {num_ok}/{len(image_paths)} images, shape={embedding.shape}")
    if dry_run:
        logger.info(f"  [DRY RUN] Would upsert {user_id} (model={model_name}, detector={detector})")
        return True
    display = display_name or user_id
    success = db_manager.upsert_enrollment(
        user_id=user_id,
        embedding=embedding,
        model=model_name,
        detector=detector,
        display_name=display,
        sync_version=1,
    )
    if success:
        logger.info(f"  SUCCESS: {user_id} enrolled")
    else:
        logger.error(f"  FAILED: DB upsert for {user_id}")
    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reprocess enrollment images to current matching schema (yolov8n + ArcFace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help=f"Directory with per-user image folders (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB),
        help=f"Target enrollment database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Config JSON to read model/detector from (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override embedding model (default: from config or ArcFace)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=None,
        help="Override detector (default: from config or yolov8n)",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Reprocess only this user (subdir name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to database",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source_dir = Path(args.source)
    db_path = Path(args.db)
    config_path = Path(args.config)

    model_name = args.model
    detector = args.detector
    if model_name is None or detector is None:
        cfg_model, cfg_detector = load_recognition_config(config_path)
        model_name = model_name or cfg_model
        detector = detector or cfg_detector

    logger.info("Reprocess enrollments -> current matching schema")
    logger.info(f"  Source:   {source_dir}")
    logger.info(f"  DB:       {db_path}")
    logger.info(f"  Model:    {model_name}")
    logger.info(f"  Detector: {detector}")
    logger.info(f"  Dry run:  {args.dry_run}")

    users = find_enrollment_images(source_dir)
    if not users:
        logger.error("No enrollment images found. Expected layout: enrolled_users/<user_id>/*.jpg or *_debug/frame_*.jpg")
        return 1

    logger.info(f"Found {len(users)} users: {list(users.keys())}")

    if args.user:
        if args.user not in users:
            logger.error(f"User '{args.user}' not found. Available: {list(users.keys())}")
            return 1
        users = {args.user: users[args.user]}

    from iot_integration.db_manager import DatabaseManager

    if not args.dry_run:
        db_manager = DatabaseManager(str(db_path), dev_mode=True)
        db_manager.initialize()
    else:
        db_manager = None

    ok = 0
    for user_id, paths in users.items():
        if reprocess_user(
            user_id=user_id,
            image_paths=paths,
            db_manager=db_manager,
            model_name=model_name,
            detector=detector,
            display_name=user_id,
            dry_run=args.dry_run,
        ):
            ok += 1

    logger.info(f"Done: {ok}/{len(users)} users processed successfully")
    return 0 if ok == len(users) else 1


if __name__ == "__main__":
    sys.exit(main())
