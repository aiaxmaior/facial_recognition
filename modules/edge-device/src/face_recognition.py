"""
Face Recognition Module

Uses ArcFace embeddings (via DeepFace) for face recognition.
Works with pre-detected faces from TRT detector.

The detection is handled by TRTFaceDetector (YOLOv8-face).
This module only handles:
- Face embedding extraction
- Embedding comparison/matching
- Enrollment database management
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Face recognition result."""
    user_id: str
    distance: float
    confidence: float  # 1 - distance
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "distance": round(self.distance, 4),
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox)
        }


class ArcFaceRecognizer:
    """
    ArcFace-based face recognition.
    
    Uses DeepFace for embedding extraction with detector_backend="skip"
    since detection is handled externally by TRTFaceDetector.
    
    Usage:
        recognizer = ArcFaceRecognizer(distance_threshold=0.55)
        recognizer.load_embeddings(db_embeddings)
        
        # face_img is a cropped face from the frame
        result = recognizer.recognize(face_img, bbox)
        if result:
            print(f"Matched: {result.user_id}")
    """
    
    def __init__(
        self,
        model_name: str = "ArcFace",
        distance_threshold: float = 0.55,
        min_face_size: int = 40
    ):
        """
        Initialize ArcFace recognizer.
        
        Args:
            model_name: DeepFace model name (ArcFace recommended)
            distance_threshold: Maximum cosine distance for match
            min_face_size: Minimum face size to process
        """
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.min_face_size = min_face_size
        
        self._embeddings_db: Dict[str, np.ndarray] = {}
        self._initialized = False
        self._deepface = None
        
        logger.info(f"ArcFaceRecognizer configured: model={model_name}, threshold={distance_threshold}")
    
    def initialize(self) -> bool:
        """
        Initialize the ArcFace model (lazy loading).
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        try:
            logger.info("Loading ArcFace model...")
            start = time.time()
            
            from deepface import DeepFace
            self._deepface = DeepFace
            
            # Warm up with dummy image
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            try:
                DeepFace.represent(
                    dummy,
                    model_name=self.model_name,
                    detector_backend="skip",
                    enforce_detection=False
                )
            except Exception as warmup_err:
                # Warmup may fail but that's ok
                logger.debug(f"Warmup note (ok to ignore): {warmup_err}")
            
            elapsed = time.time() - start
            logger.info(f"ArcFace model loaded in {elapsed:.1f}s")
            self._initialized = True
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to load ArcFace model: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def load_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        """
        Load enrollment embeddings into memory.
        
        Args:
            embeddings: Dict mapping user_id to embedding numpy array
        """
        self._embeddings_db = {}
        
        for user_id, emb in embeddings.items():
            # Ensure embeddings are normalized for cosine similarity
            if isinstance(emb, np.ndarray):
                norm = np.linalg.norm(emb)
                if norm > 0:
                    self._embeddings_db[user_id] = emb / norm
                else:
                    self._embeddings_db[user_id] = emb
            else:
                self._embeddings_db[user_id] = np.array(emb)
        
        logger.info(f"Loaded {len(self._embeddings_db)} embeddings")
    
    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ArcFace embedding from face image.
        
        Args:
            face_img: BGR face image (cropped, any size)
            
        Returns:
            Normalized embedding vector or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            result = self._deepface.represent(
                face_img,
                model_name=self.model_name,
                detector_backend="skip",  # Face already detected
                enforce_detection=False
            )
            
            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"])
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    return embedding / norm
                return embedding
            
            return None
            
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None
    
    def find_match(
        self,
        embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find closest matches for an embedding.
        
        Args:
            embedding: Query embedding (normalized)
            top_k: Number of top matches to return
            
        Returns:
            List of (user_id, distance) tuples, sorted by distance
        """
        if not self._embeddings_db:
            return []
        
        distances = []
        
        for user_id, db_emb in self._embeddings_db.items():
            # Cosine distance = 1 - cosine_similarity
            # For normalized vectors: cosine_sim = dot product
            sim = np.dot(embedding, db_emb)
            distance = 1.0 - sim
            distances.append((user_id, distance))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Debug: show ALL candidates periodically
        if not hasattr(self, '_match_count'):
            self._match_count = 0
        self._match_count += 1
        if self._match_count <= 3:
            logger.info(f"[CANDIDATES] ALL {len(distances)}: {[(u, f'{d:.4f}') for u, d in distances]}")
        
        return distances[:top_k]
    
    def recognize(
        self,
        face_img: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[RecognitionResult]:
        """
        Recognize a face against enrolled embeddings.
        
        Args:
            face_img: Cropped face image (BGR)
            bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            RecognitionResult if match found, None otherwise
        """
        # Check minimum size
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < self.min_face_size or h < self.min_face_size:
            logger.debug(f"Face too small: {w}x{h}")
            return None
        
        # Get embedding
        embedding = self.get_embedding(face_img)
        if embedding is None:
            return None
        
        # Find best match
        matches = self.find_match(embedding, top_k=1)
        if not matches:
            return None
        
        best_user_id, best_distance = matches[0]
        
        # Check threshold
        if best_distance <= self.distance_threshold:
            logger.info(f"[MATCH] {best_user_id} (distance={best_distance:.4f})")
            return RecognitionResult(
                user_id=best_user_id,
                distance=best_distance,
                confidence=1.0 - best_distance,
                bbox=bbox,
                embedding=embedding
            )
        
        logger.debug(f"[NO MATCH] Best: {best_user_id} at {best_distance:.4f} > threshold {self.distance_threshold}")
        return None
    
    def recognize_batch(
        self,
        faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
    ) -> List[Optional[RecognitionResult]]:
        """
        Recognize multiple faces.
        
        Args:
            faces: List of (face_img, bbox) tuples
            
        Returns:
            List of RecognitionResult or None for each face
        """
        return [self.recognize(img, bbox) for img, bbox in faces]
    
    @property
    def num_enrolled(self) -> int:
        """Number of enrolled faces."""
        return len(self._embeddings_db)
    
    @property
    def enrolled_users(self) -> List[str]:
        """List of enrolled user IDs."""
        return list(self._embeddings_db.keys())


# Test
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ArcFace Recognizer...")
    
    recognizer = ArcFaceRecognizer(distance_threshold=0.55)
    
    # Initialize
    if not recognizer.initialize():
        print("Failed to initialize")
        sys.exit(1)
    
    # Test embedding extraction
    print("\nExtracting embedding from dummy image...")
    dummy = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    start = time.time()
    embedding = recognizer.get_embedding(dummy)
    elapsed = (time.time() - start) * 1000
    
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"Extraction time: {elapsed:.1f}ms")
    else:
        print("Embedding extraction failed (expected for random image)")
    
    # Benchmark
    print("\nBenchmarking embedding extraction (10 iterations)...")
    times = []
    for _ in range(10):
        start = time.time()
        recognizer.get_embedding(dummy)
        times.append((time.time() - start) * 1000)
    
    print(f"Average: {np.mean(times):.1f}ms")
    print(f"Min: {np.min(times):.1f}ms")
    print(f"Max: {np.max(times):.1f}ms")
