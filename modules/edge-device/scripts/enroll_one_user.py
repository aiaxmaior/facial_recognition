#!/usr/bin/env python3
"""
Enroll a single user into the facial recognition DB.

Use this to add one person at a time so you can verify who is enrolled.
Uses the same model/detector as the recognition pipeline (yolov8n + ArcFace by default).

Usage:
    # One image
    python scripts/enroll_one_user.py --user-id user_3 --image path/to/face.jpg

    # Multiple images for this user (embedding = average, or first only)
    python scripts/enroll_one_user.py --user-id user_3 --dir ~/facial_recognition/enrolled_users/Ryan_Larsen_debug
    python scripts/enroll_one_user.py --user-id user_3 --images img1.jpg img2.jpg

    # With display name
    python scripts/enroll_one_user.py --user-id user_3 --name "Ryan Larsen" --image face.jpg

    # Use first image only when passing a directory (no averaging)
    python scripts/enroll_one_user.py --user-id user_3 --dir path/to/photos --first-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DB = PROJECT_DIR / "data" / "enrollments.db"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "config.json"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "yolov8n"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(config_path: Path) -> tuple:
    """Return (model, detector) from config JSON."""
    if not config_path.exists():
        return MODEL_NAME, DETECTOR_BACKEND
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        rec = cfg.get("recognition", {})
        return rec.get("model", MODEL_NAME), rec.get("detector_backend", DETECTOR_BACKEND)
    except Exception as e:
        logger.warning(f"Could not read config: {e}")
        return MODEL_NAME, DETECTOR_BACKEND


def collect_images(args) -> List[Path]:
    """Resolve one or more image paths from --image, --images, or --dir."""
    base = PROJECT_DIR
    out: List[Path] = []

    if args.image:
        p = Path(args.image)
        if not p.is_absolute():
            p = base / p
        if p.exists():
            out.append(p)
        else:
            logger.error(f"Image not found: {p}")
            return []
    if args.images:
        for s in args.images:
            p = Path(s)
            if not p.is_absolute():
                p = base / p
            if p.exists():
                out.append(p)
            else:
                logger.warning(f"Image not found: {p}")
    if args.dir:
        d = Path(args.dir)
        if not d.is_absolute():
            d = base / d
        if not d.is_dir():
            logger.error(f"Directory not found: {d}")
            return []
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                out.append(f)

    return out


def extract_embedding(image_path: Path, model_name: str, detector: str) -> Optional[np.ndarray]:
    """Extract one face embedding from an image."""
    try:
        from deepface import DeepFace
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read: {image_path}")
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
        logger.warning(f"Embedding failed for {image_path}: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enroll one user into the recognition DB (one user per run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--user-id", required=True, help="User ID to store (e.g. user_3, ryan_larsen)")
    parser.add_argument("--name", default=None, help="Display name (default: same as user-id)")
    parser.add_argument("--image", type=str, help="Path to a single face image")
    parser.add_argument("--images", nargs="+", help="Paths to multiple images for this user")
    parser.add_argument("--dir", type=str, help="Directory of images for this user")
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="With --dir or --images: use only the first image (no averaging)",
    )
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="Path to enrollments DB")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Config JSON for model/detector")
    parser.add_argument("--model", type=str, default=None, help="Override model (default: from config)")
    parser.add_argument("--detector", type=str, default=None, help="Override detector (default: from config)")
    args = parser.parse_args()

    # Require at least one image source
    if not args.image and not args.images and not args.dir:
        logger.error("Provide one of: --image, --images, or --dir")
        return 1

    paths = collect_images(args)
    if not paths:
        logger.error("No valid image paths found")
        return 1

    logger.info(f"User ID: {args.user_id}")
    logger.info(f"Display name: {args.name or args.user_id}")
    logger.info(f"Images: {len(paths)} file(s)")

    model_name = args.model
    detector = args.detector
    if model_name is None or detector is None:
        m, d = load_config(Path(args.config))
        model_name = model_name or m
        detector = detector or d

    logger.info(f"Model: {model_name}, Detector: {detector}")

    # Build embedding: single image or first-only => one; else average
    if len(paths) == 1 or args.first_only:
        use_paths = [paths[0]]
        logger.info(f"Using single image: {use_paths[0]}")
    else:
        use_paths = paths
        logger.info(f"Averaging embeddings from {len(use_paths)} images")

    embeddings = []
    for p in use_paths:
        emb = extract_embedding(p, model_name, detector)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        logger.error("No embeddings extracted from any image")
        return 1

    if len(embeddings) == 1:
        embedding = embeddings[0]
    else:
        embedding = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

    logger.info(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_DIR / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from iot_integration.db_manager import DatabaseManager

    db = DatabaseManager(str(db_path), dev_mode=True)
    db.initialize()

    display_name = args.name or args.user_id
    success = db.upsert_enrollment(
        user_id=args.user_id,
        embedding=embedding,
        model=model_name,
        detector=detector,
        display_name=display_name,
        sync_version=1,
    )

    if success:
        logger.info(f"Enrolled: user_id={args.user_id}, name={display_name}")
        count = db.get_enrollment_count()
        logger.info(f"Total users in DB: {count}")
        return 0
    logger.error("Database upsert failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
