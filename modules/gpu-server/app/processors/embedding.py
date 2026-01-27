"""
Facial Embedding Processor

Uses DeepFace with ArcFace model to generate facial embeddings.
"""

import base64
import io
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

import numpy as np
from PIL import Image


class EmbeddingProcessor:
    """
    Facial embedding processor using DeepFace/ArcFace.
    
    Generates 512-dimensional embeddings for facial recognition.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model_name", "ArcFace")
        self.detector_backend = config.get("detector_backend", "yolov8")
        self.embedding_dim = config.get("embedding_dim", 512)
        self.enforce_detection = config.get("enforce_detection", True)
        self.align = config.get("align", True)
        
        self._deepface = None
        self.is_ready = False
    
    async def initialize(self):
        """Initialize the DeepFace model"""
        try:
            # Run in thread pool to not block async
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            self.is_ready = True
            logger.info(f"EmbeddingProcessor initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingProcessor: {e}")
            # Fall back to mock mode
            self.is_ready = True
            logger.warning("EmbeddingProcessor running in mock mode")
    
    def _load_model(self):
        """Load DeepFace model (runs in thread)"""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            # Pre-load model by running a dummy embedding
            logger.debug(f"Pre-loading {self.model_name} model...")
            # Model will be loaded on first use
        except ImportError:
            logger.warning("DeepFace not installed, running in mock mode")
            self._deepface = None
    
    async def cleanup(self):
        """Cleanup resources"""
        self.is_ready = False
        logger.info("EmbeddingProcessor cleaned up")
    
    async def process(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]],
        options: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Process images and generate facial embedding.
        
        Args:
            employee_id: Employee identifier
            images: List of {"pose": str, "bytes": bytes}
            options: Optional processing options
            
        Returns:
            {
                "embedding_base64": str,
                "embedding_dim": int,
                "model": str,
                "thumbnail_base64": str
            }
        """
        logger.debug(f"Processing {len(images)} images for employee {employee_id}")
        
        if self._deepface is None:
            # Mock mode
            return self._mock_process(employee_id, images)
        
        # Run processing in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._process_sync, 
            employee_id, 
            images, 
            options
        )
        
        return result
    
    def _process_sync(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]],
        options: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Synchronous processing (runs in thread)"""
        embeddings = []
        front_image = None
        
        for img_data in images:
            pose = img_data.get("pose", "unknown")
            img_bytes = img_data["bytes"]
            
            # Convert bytes to numpy array
            pil_image = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(pil_image)
            
            # Keep front image for thumbnail
            if pose == "front" or front_image is None:
                front_image = pil_image.copy()
            
            try:
                # Generate embedding
                result = self._deepface.represent(
                    img_path=img_array,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=self.enforce_detection,
                    align=self.align
                )
                
                if result and len(result) > 0:
                    embedding = result[0]["embedding"]
                    embeddings.append(embedding)
                    logger.debug(f"Generated embedding for pose '{pose}': dim={len(embedding)}")
                    
            except Exception as e:
                logger.warning(f"Failed to process pose '{pose}': {e}")
                continue
        
        if not embeddings:
            raise ValueError("No faces detected in any of the provided images")
        
        # Average embeddings from multiple images
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        
        # Normalize embedding
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        # Convert to base64
        embedding_base64 = base64.b64encode(avg_embedding.tobytes()).decode('utf-8')
        
        # Generate thumbnail from front image
        thumbnail_base64 = self._generate_thumbnail(front_image)
        
        return {
            "embedding_base64": embedding_base64,
            "embedding_dim": len(avg_embedding),
            "model": self.model_name,
            "thumbnail_base64": thumbnail_base64
        }
    
    def _generate_thumbnail(self, image: Image.Image, size: tuple = (128, 128)) -> str:
        """Generate 128x128 thumbnail from image"""
        try:
            # Try to extract face region for better thumbnail
            if self._deepface:
                img_array = np.array(image)
                try:
                    faces = self._deepface.extract_faces(
                        img_path=img_array,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        align=True
                    )
                    if faces and len(faces) > 0:
                        face_img = faces[0]["face"]
                        # Convert from float [0,1] to uint8 [0,255]
                        face_img = (face_img * 255).astype(np.uint8)
                        image = Image.fromarray(face_img)
                except Exception as e:
                    logger.debug(f"Face extraction for thumbnail failed, using full image: {e}")
            
            # Resize to thumbnail size
            image = image.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create square canvas
            canvas = Image.new("RGB", size, (255, 255, 255))
            offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
            canvas.paste(image, offset)
            
            # Convert to base64 JPEG
            buffer = io.BytesIO()
            canvas.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            # Return placeholder
            return ""
    
    def _mock_process(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Mock processing for testing without DeepFace"""
        logger.debug(f"Mock processing for employee {employee_id}")
        
        # Generate random embedding
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        embedding_base64 = base64.b64encode(embedding.tobytes()).decode('utf-8')
        
        # Generate thumbnail from first image
        thumbnail_base64 = ""
        if images:
            try:
                pil_image = Image.open(io.BytesIO(images[0]["bytes"]))
                thumbnail_base64 = self._generate_thumbnail(pil_image)
            except Exception as e:
                logger.warning(f"Mock thumbnail generation failed: {e}")
        
        return {
            "embedding_base64": embedding_base64,
            "embedding_dim": self.embedding_dim,
            "model": f"{self.model_name} (mock)",
            "thumbnail_base64": thumbnail_base64
        }
