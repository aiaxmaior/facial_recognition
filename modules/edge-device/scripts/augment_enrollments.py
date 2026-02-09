#!/usr/bin/env python3
"""
Augment enrollment images for more robust ArcFace embeddings.

For each base enrollment image (frame_1.jpg ... frame_5.jpg), generates
photometric augmentation variants that simulate real RTSP camera conditions.

Augmentations applied per base image:
  1. bright         — +25% brightness (daytime / direct lighting)
  2. dark_noisy     — -20% brightness + Gaussian noise σ=12 (low-light sensor noise)
  3. blur_shift     — Gaussian blur 5×5 + ±5% random crop (motion blur + bbox jitter)
  4. warm           — hue +8°, saturation +12% (white balance / color temp drift)
  5. compressed     — JPEG re-encode at quality 50 (RTSP H.264 stream artifacts)

Result: 5 originals + 25 augmented = 30 images per user.

After augmenting, re-run the enrollment script to recompute averaged embeddings:
    python scripts/enroll_from_naming_schema.py

Usage:
    python scripts/augment_enrollments.py                         # augment all users
    python scripts/augment_enrollments.py --preview               # show sample, don't save
    python scripts/augment_enrollments.py --clean                 # remove all augmented images
    python scripts/augment_enrollments.py --source enrolled_users # custom source dir
    python scripts/augment_enrollments.py --user Ryan_Larsen_debug  # single user only
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SOURCE = PROJECT_DIR / "enrolled_users"

# Augmented file marker — used for identification and cleanup
AUG_TAG = "_aug_"

# Base images are named frame_N.jpg (N=1..5)
BASE_PATTERN = "frame_*.jpg"

# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------
# Each takes a BGR image (np.ndarray) and returns a BGR image (np.ndarray).
# Deterministic given the same input (no internal randomness).
# The blur_shift augmentation uses a fixed crop offset derived from image dims.
# ---------------------------------------------------------------------------


def aug_bright(img: np.ndarray) -> np.ndarray:
    """Increase brightness by ~25% (simulates bright / direct lighting)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_dark_noisy(img: np.ndarray) -> np.ndarray:
    """Decrease brightness 20% and add Gaussian sensor noise σ=12."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.80, 0, 255)
    dark = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    noise = np.random.default_rng(42).normal(0, 12, dark.shape).astype(np.float32)
    noisy = np.clip(dark.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def aug_blur_shift(img: np.ndarray) -> np.ndarray:
    """Gaussian blur 5×5 + deterministic 5% crop (simulates motion blur + bbox jitter)."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    h, w = blurred.shape[:2]
    # Deterministic 5% crop from top-left offset
    dx = max(1, int(w * 0.05))
    dy = max(1, int(h * 0.05))
    cropped = blurred[dy : h - dy // 2, dx : w - dx // 2]
    # Resize back to original dimensions to keep embedding input consistent
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def aug_warm(img: np.ndarray) -> np.ndarray:
    """Warm color temperature shift: hue +8°, saturation +12%."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 8) % 180  # OpenCV hue range is 0-179
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_compressed(img: np.ndarray) -> np.ndarray:
    """JPEG re-compression at quality 50 (simulates RTSP stream artifacts)."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# Ordered list of (suffix, function) — suffix becomes part of the filename
AUGMENTATIONS: List[Tuple[str, callable]] = [
    ("bright", aug_bright),
    ("dark_noisy", aug_dark_noisy),
    ("blur_shift", aug_blur_shift),
    ("warm", aug_warm),
    ("compressed", aug_compressed),
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_base_images(user_dir: Path) -> List[Path]:
    """Find original (non-augmented) frame images in a user directory."""
    all_frames = sorted(user_dir.glob(BASE_PATTERN))
    return [p for p in all_frames if AUG_TAG not in p.name]


def find_augmented_images(user_dir: Path) -> List[Path]:
    """Find augmented images in a user directory."""
    all_frames = sorted(user_dir.glob(BASE_PATTERN))
    return [p for p in all_frames if AUG_TAG in p.name]


def find_user_dirs(source_dir: Path, user_filter: Optional[str] = None) -> List[Path]:
    """Find all user enrollment directories."""
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return []
    dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if user_filter:
        dirs = [d for d in dirs if d.name == user_filter]
        if not dirs:
            logger.error(f"User directory not found: {user_filter}")
    return dirs


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def augment_user(user_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Generate augmented images for a single user.

    Returns:
        (num_augmented, num_skipped) — skipped means the file already exists.
    """
    base_images = find_base_images(user_dir)
    if not base_images:
        logger.warning(f"  No base images found in {user_dir.name}")
        return 0, 0

    num_created = 0
    num_skipped = 0

    for base_path in base_images:
        img = cv2.imread(str(base_path))
        if img is None:
            logger.warning(f"  Failed to read {base_path.name}")
            continue

        stem = base_path.stem  # e.g. "frame_1"

        for suffix, aug_fn in AUGMENTATIONS:
            aug_name = f"{stem}{AUG_TAG}{suffix}.jpg"
            aug_path = user_dir / aug_name

            if aug_path.exists():
                num_skipped += 1
                continue

            if dry_run:
                logger.info(f"  [DRY RUN] Would create {aug_name}")
                num_created += 1
                continue

            aug_img = aug_fn(img)
            cv2.imwrite(str(aug_path), aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            num_created += 1

    return num_created, num_skipped


def clean_user(user_dir: Path, dry_run: bool = False) -> int:
    """Remove all augmented images for a single user. Returns count removed."""
    aug_images = find_augmented_images(user_dir)
    if not aug_images:
        return 0

    removed = 0
    for p in aug_images:
        if dry_run:
            logger.info(f"  [DRY RUN] Would remove {p.name}")
        else:
            p.unlink()
        removed += 1

    return removed


def preview_user(user_dir: Path) -> None:
    """Generate and display a sample augmentation grid for visual inspection."""
    base_images = find_base_images(user_dir)
    if not base_images:
        logger.warning(f"  No base images found in {user_dir.name}")
        return

    # Use the first base image as the sample
    sample_path = base_images[0]
    img = cv2.imread(str(sample_path))
    if img is None:
        logger.error(f"  Cannot read {sample_path}")
        return

    # Build a grid: original + all augmentations
    thumb_w, thumb_h = 320, 240
    labels = ["ORIGINAL"] + [name.upper() for name, _ in AUGMENTATIONS]
    images = [img] + [fn(img) for _, fn in AUGMENTATIONS]

    cols = 3
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * (thumb_h + 30), cols * thumb_w, 3), dtype=np.uint8)

    for idx, (label, aug_img) in enumerate(zip(labels, images)):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(aug_img, (thumb_w, thumb_h))
        y_off = r * (thumb_h + 30)
        x_off = c * thumb_w
        grid[y_off : y_off + thumb_h, x_off : x_off + thumb_w] = thumb
        # Label
        cv2.putText(
            grid, label, (x_off + 5, y_off + thumb_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

    # Save preview grid
    preview_dir = PROJECT_DIR / "data"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"augmentation_preview_{user_dir.name}.jpg"
    cv2.imwrite(str(preview_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    logger.info(f"  Preview saved: {preview_path}")

    # Try to show the window (works with X11 forwarding or local display)
    try:
        title = f"Augmentation Preview - {user_dir.name}"
        cv2.imshow(title, grid)
        logger.info(f"  Showing preview window (press any key to continue)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        logger.info(f"  No display available — preview image saved to {preview_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augment enrollment images for robust ArcFace embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "After augmenting, re-run enrollment to recompute embeddings:\n"
            "  python scripts/enroll_from_naming_schema.py\n"
        ),
    )
    parser.add_argument(
        "--source", default=str(DEFAULT_SOURCE),
        help="Path to enrolled_users directory (default: %(default)s)",
    )
    parser.add_argument(
        "--user", default=None,
        help="Process only this user directory name (e.g. Ryan_Larsen_debug)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove all augmented images instead of creating them",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Generate and show a sample augmentation grid (first user, first image)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = PROJECT_DIR / source_dir

    user_dirs = find_user_dirs(source_dir, args.user)
    if not user_dirs:
        logger.error("No user directories found")
        return 1

    logger.info(f"Source: {source_dir}")
    logger.info(f"Users:  {len(user_dirs)}")
    logger.info(f"Augmentations per base image: {len(AUGMENTATIONS)} "
                f"({', '.join(name for name, _ in AUGMENTATIONS)})")

    # --- Preview mode ---
    if args.preview:
        logger.info(f"\n--- Preview mode ---")
        target = user_dirs[0]
        logger.info(f"Generating preview for: {target.name}")
        preview_user(target)
        return 0

    # --- Clean mode ---
    if args.clean:
        logger.info(f"\n--- Clean mode {'(dry run)' if args.dry_run else ''} ---")
        total_removed = 0
        for user_dir in user_dirs:
            count = clean_user(user_dir, dry_run=args.dry_run)
            if count:
                logger.info(f"  {user_dir.name}: removed {count} augmented images")
            total_removed += count
        logger.info(f"Total removed: {total_removed}")
        return 0

    # --- Augment mode (default) ---
    logger.info(f"\n--- Augment mode {'(dry run)' if args.dry_run else ''} ---")
    total_created = 0
    total_skipped = 0

    for user_dir in user_dirs:
        base_count = len(find_base_images(user_dir))
        existing_aug = len(find_augmented_images(user_dir))

        logger.info(f"Processing: {user_dir.name} "
                     f"({base_count} base, {existing_aug} existing augmented)")

        created, skipped = augment_user(user_dir, dry_run=args.dry_run)
        total_count = base_count + existing_aug + created

        logger.info(f"  Created: {created}, Skipped (exist): {skipped}, "
                     f"Total images: {total_count}")

        total_created += created
        total_skipped += skipped

    logger.info(f"\nDone: {total_created} created, {total_skipped} skipped (already exist)")

    if total_created > 0 and not args.dry_run:
        logger.info(
            "\nNext step — re-run enrollment to recompute averaged embeddings:\n"
            "  python scripts/enroll_from_naming_schema.py"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
