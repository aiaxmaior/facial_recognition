#!/usr/bin/env python3
"""
Auto-process enrollments from folder naming schema with dummy IDs.

Scans enrolled_users (e.g. Ryan_Larsen_debug/, Arjun_Joshi_test/), derives
display names from folder names (e.g. "Ryan Larsen", "Arjun Joshi"), assigns
dummy user_ids (user_001, user_002, ...), and upserts embeddings so you can
test recognition by name.

Usage:
    python scripts/enroll_from_naming_schema.py
    python scripts/enroll_from_naming_schema.py --source enrolled_users --db data/enrollments.db
    python scripts/enroll_from_naming_schema.py --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SOURCE = PROJECT_DIR / "enrolled_users"
DEFAULT_DB = PROJECT_DIR / "data" / "enrollments.db"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "config.json"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "yolov8n"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def folder_to_display_name(folder_name: str) -> str:
    """Derive display name from folder: strip _debug/_test, replace _ with space, title case."""
    s = folder_name
    for suffix in ("_debug", "_test"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]
            break
    s = s.replace("_", " ").strip()
    return s.title() if s else folder_name


def load_config(config_path: Path) -> Tuple[str, str]:
    if not config_path.exists():
        return MODEL_NAME, DETECTOR_BACKEND
    try:
        with open(config_path) as f:
            rec = json.load(f).get("recognition", {})
        return rec.get("model", MODEL_NAME), rec.get("detector_backend", DETECTOR_BACKEND)
    except Exception as e:
        logger.warning(f"Config: {e}")
        return MODEL_NAME, DETECTOR_BACKEND


def find_enrollment_images(base_dir: Path) -> Dict[str, List[Path]]:
    users: Dict[str, List[Path]] = {}
    if not base_dir.exists():
        logger.error(f"Source not found: {base_dir}")
        return users
    for debug_dir in base_dir.glob("*_debug"):
        if not debug_dir.is_dir():
            continue
        images = sorted(debug_dir.glob("frame_*.jpg"))
        if images:
            users[debug_dir.name] = images
    for subdir in base_dir.iterdir():
        if not subdir.is_dir() or subdir.name.endswith("_debug"):
            continue
        images = [p for p in subdir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            users[subdir.name] = sorted(images)
    return users


def extract_embedding(image_path: Path, model_name: str, detector: str) -> Optional[np.ndarray]:
    try:
        from deepface import DeepFace
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        result = DeepFace.represent(
            img, model_name=model_name, detector_backend=detector,
            enforce_detection=False, align=True,
        )
        if result and len(result) > 0:
            return np.array(result[0]["embedding"], dtype=np.float32)
        return None
    except Exception as e:
        logger.warning(f"Embedding failed {image_path}: {e}")
        return None


def compute_average_embedding(
    image_paths: List[Path], model_name: str, detector: str
) -> Tuple[Optional[np.ndarray], int]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll from naming schema with dummy IDs")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="enrolled_users dir")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="enrollments DB")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config JSON")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = PROJECT_DIR / source_dir
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_DIR / db_path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path

    model_name, detector = load_config(config_path)
    users = find_enrollment_images(source_dir)
    if not users:
        logger.error("No enrollment folders found")
        return 1

    # Stable order and dummy IDs
    folder_names = sorted(users.keys())
    dummy_ids = [f"user_{i:03d}" for i in range(1, len(folder_names) + 1)]

    logger.info(f"Source: {source_dir}, DB: {db_path}, model={model_name}, detector={detector}")
    logger.info(f"Found {len(users)} folders; dummy IDs user_001..user_{len(users):03d}")

    from iot_integration.db_manager import DatabaseManager
    db = DatabaseManager(str(db_path), dev_mode=True)
    db.initialize()

    ok = 0
    for folder_name, dummy_id in zip(folder_names, dummy_ids):
        paths = users[folder_name]
        display_name = folder_to_display_name(folder_name)
        logger.info(f"Processing {dummy_id} <- {folder_name} (display: {display_name!r})")
        embedding, num_ok = compute_average_embedding(paths, model_name, detector)
        if embedding is None:
            logger.error(f"  FAILED: no embeddings for {folder_name}")
            continue
        logger.info(f"  Embedding from {num_ok}/{len(paths)} images")
        if args.dry_run:
            logger.info(f"  [DRY RUN] Would upsert {dummy_id} display_name={display_name!r}")
            ok += 1
            continue
        success = db.upsert_enrollment(
            user_id=dummy_id,
            embedding=embedding,
            model=model_name,
            detector=detector,
            display_name=display_name,
            sync_version=1,
        )
        if success:
            logger.info(f"  Enrolled: {dummy_id} -> {display_name!r}")
            ok += 1
        else:
            logger.error(f"  DB upsert failed for {dummy_id}")

    logger.info(f"Done: {ok}/{len(users)} enrolled")
    return 0 if ok == len(users) else 1


if __name__ == "__main__":
    sys.exit(main())
