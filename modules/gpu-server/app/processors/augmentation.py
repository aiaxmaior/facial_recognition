"""
Enrollment-image augmentation for robust ArcFace embeddings.

For each base enrollment image the following photometric / geometric
variants are produced to simulate real RTSP-camera conditions:

  1. bright         — +25 % brightness  (daytime / direct lighting)
  2. dark_noisy     — -20 % brightness + Gaussian noise σ=12  (low-light sensor noise)
  3. blur_shift     — Gaussian blur 5×5 + 5 % crop  (motion blur + bbox jitter)
  4. warm           — hue +8°, saturation +12 %  (white-balance / colour-temp drift)
  5. compressed     — JPEG re-encode @ quality 50  (RTSP H.264 stream artefacts)

With 5 base images this yields  5 originals + 25 augmented = **30 images**
that are all embedded and averaged into a single enrollment vector.
"""

from typing import Callable, List, Tuple

import cv2
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------
# Each takes a BGR numpy array and returns a BGR numpy array.

def aug_bright(img: np.ndarray) -> np.ndarray:
    """Increase brightness by ~25 % (simulates bright / direct lighting)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_dark_noisy(img: np.ndarray) -> np.ndarray:
    """Decrease brightness 20 % and add Gaussian sensor noise σ=12."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.80, 0, 255)
    dark = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    noise = np.random.default_rng(42).normal(0, 12, dark.shape).astype(np.float32)
    return np.clip(dark.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_blur_shift(img: np.ndarray) -> np.ndarray:
    """Gaussian blur 5×5 + deterministic 5 % crop (motion blur + bbox jitter)."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    h, w = blurred.shape[:2]
    dx = max(1, int(w * 0.05))
    dy = max(1, int(h * 0.05))
    cropped = blurred[dy : h - dy // 2, dx : w - dx // 2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def aug_warm(img: np.ndarray) -> np.ndarray:
    """Warm colour-temperature shift: hue +8°, saturation +12 %."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 8) % 180  # OpenCV hue range 0-179
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_compressed(img: np.ndarray) -> np.ndarray:
    """JPEG re-compression at quality 50 (simulates RTSP stream artefacts)."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# Ordered registry: (human-readable name, function)
AUGMENTATIONS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("bright", aug_bright),
    ("dark_noisy", aug_dark_noisy),
    ("blur_shift", aug_blur_shift),
    ("warm", aug_warm),
    ("compressed", aug_compressed),
]


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def augment_image(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate all augmented variants for a single BGR image.

    Returns a list of ``(label, augmented_image)`` tuples.
    The *original* image is **not** included in the output — the caller
    should prepend it if needed.
    """
    results: List[Tuple[str, np.ndarray]] = []
    for name, fn in AUGMENTATIONS:
        try:
            aug = fn(img)
            results.append((name, aug))
        except Exception as e:
            logger.warning(f"Augmentation '{name}' failed, skipping: {e}")
    return results


def augment_batch(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Expand a list of BGR images into originals + all augmentations.

    Given *N* input images the output contains up to ``N * 6`` images
    (each original followed by its 5 augmented variants).

    This is the convenience function used by :class:`EmbeddingProcessor`.
    """
    expanded: List[np.ndarray] = []
    for idx, img in enumerate(images):
        expanded.append(img)  # original
        for name, aug_img in augment_image(img):
            expanded.append(aug_img)
        logger.debug(
            f"Image {idx + 1}/{len(images)}: original + "
            f"{len(AUGMENTATIONS)} augmentations"
        )
    logger.info(
        f"Augmentation complete: {len(images)} originals → "
        f"{len(expanded)} total images"
    )
    return expanded
