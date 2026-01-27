"""
Emotion Analysis Processor

Uses VLM or DeepFace for emotion analysis in images/video frames.
"""

import base64
import io
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from loguru import logger

import numpy as np
from PIL import Image


class EmotionProcessor:
    """
    Emotion analysis processor.
    
    Supports multiple backends:
    - vllm: Use a VLM endpoint for emotion analysis
    - deepface: Use DeepFace emotion detection
    - mock: Return mock data for testing
    """
    
    EMOTIONS = ["neutral", "happiness", "sadness", "anger", "fear", "surprise", "disgust", "contempt"]
    
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("backend", "mock")
        
        # VLM settings
        self.llm_base_url = config.get("llm_base_url")
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.max_tokens = config.get("max_tokens", 256)
        self.temperature = config.get("temperature", 0.1)
        
        self._deepface = None
        self._http_client = None
        self.is_ready = False
    
    async def initialize(self):
        """Initialize the emotion processor"""
        try:
            if self.backend == "vllm":
                await self._init_vllm()
            elif self.backend == "deepface":
                await self._init_deepface()
            else:
                # Mock mode
                logger.info("EmotionProcessor running in mock mode")
            
            self.is_ready = True
            logger.info(f"EmotionProcessor initialized with backend: {self.backend}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmotionProcessor: {e}")
            self.backend = "mock"
            self.is_ready = True
            logger.warning("EmotionProcessor falling back to mock mode")
    
    async def _init_vllm(self):
        """Initialize VLM HTTP client"""
        self._http_client = httpx.AsyncClient(timeout=60.0)
        
        # Test connection if base URL is configured
        if self.llm_base_url:
            try:
                response = await self._http_client.get(f"{self.llm_base_url}/health")
                logger.info(f"VLM endpoint reachable: {self.llm_base_url}")
            except Exception as e:
                logger.warning(f"VLM endpoint not reachable: {e}")
    
    async def _init_deepface(self):
        """Initialize DeepFace for emotion detection"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_deepface)
    
    def _load_deepface(self):
        """Load DeepFace (runs in thread)"""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            logger.info("DeepFace loaded for emotion analysis")
        except ImportError:
            logger.warning("DeepFace not installed")
            self._deepface = None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._http_client:
            await self._http_client.aclose()
        self.is_ready = False
        logger.info("EmotionProcessor cleaned up")
    
    async def process(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]],
        options: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze emotions in images.
        
        Args:
            employee_id: Employee identifier
            images: List of {"frame": int, "bytes": bytes}
            options: Optional processing options
            
        Returns:
            {
                "emotions": [{"frame": int, "emotion": str, "confidence": float, ...}],
                "dominant_emotion": str,
                "summary": str (optional)
            }
        """
        logger.debug(f"Analyzing emotions in {len(images)} frames for employee {employee_id}")
        
        if self.backend == "vllm" and self._http_client:
            return await self._process_vllm(employee_id, images, options)
        elif self.backend == "deepface" and self._deepface:
            return await self._process_deepface(employee_id, images, options)
        else:
            return self._mock_process(employee_id, images)
    
    async def _process_vllm(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]],
        options: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Process using VLM endpoint"""
        emotions = []
        
        for img_data in images:
            frame = img_data.get("frame", 0)
            img_bytes = img_data["bytes"]
            
            # Convert to base64 for VLM
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Create VLM prompt
            prompt = """Analyze the emotion expressed in this image. 
            Return a JSON object with:
            - emotion: one of [neutral, happiness, sadness, anger, fear, surprise, disgust, contempt]
            - confidence: a float between 0 and 1
            - description: brief description of the emotional expression
            
            Only return the JSON object, no other text."""
            
            try:
                response = await self._http_client.post(
                    f"{self.llm_base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name or "default",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                                ]
                            }
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    
                    # Parse JSON from response
                    import json
                    try:
                        emotion_data = json.loads(content)
                        emotions.append({
                            "frame": frame,
                            "emotion": emotion_data.get("emotion", "neutral"),
                            "confidence": emotion_data.get("confidence", 0.5),
                            "description": emotion_data.get("description", "")
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse VLM response for frame {frame}")
                        emotions.append({"frame": frame, "emotion": "neutral", "confidence": 0.0})
                else:
                    logger.warning(f"VLM request failed for frame {frame}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"VLM processing error for frame {frame}: {e}")
                emotions.append({"frame": frame, "emotion": "neutral", "confidence": 0.0, "error": str(e)})
        
        # Determine dominant emotion
        emotion_counts = {}
        for e in emotions:
            em = e.get("emotion", "neutral")
            emotion_counts[em] = emotion_counts.get(em, 0) + e.get("confidence", 1)
        
        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        
        return {
            "emotions": emotions,
            "dominant_emotion": dominant,
            "summary": f"Analyzed {len(images)} frames. Dominant emotion: {dominant}"
        }
    
    async def _process_deepface(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]],
        options: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Process using DeepFace emotion detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._process_deepface_sync, 
            employee_id, 
            images, 
            options
        )
    
    def _process_deepface_sync(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]],
        options: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Synchronous DeepFace processing (runs in thread)"""
        emotions = []
        
        for img_data in images:
            frame = img_data.get("frame", 0)
            img_bytes = img_data["bytes"]
            
            # Convert bytes to numpy array
            pil_image = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(pil_image)
            
            try:
                result = self._deepface.analyze(
                    img_path=img_array,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                
                if result and len(result) > 0:
                    emotion_scores = result[0].get("emotion", {})
                    dominant = result[0].get("dominant_emotion", "neutral")
                    confidence = emotion_scores.get(dominant, 0) / 100.0
                    
                    emotions.append({
                        "frame": frame,
                        "emotion": dominant.lower(),
                        "confidence": confidence,
                        "all_emotions": {k.lower(): v / 100.0 for k, v in emotion_scores.items()}
                    })
                else:
                    emotions.append({"frame": frame, "emotion": "neutral", "confidence": 0.0})
                    
            except Exception as e:
                logger.warning(f"DeepFace emotion analysis failed for frame {frame}: {e}")
                emotions.append({"frame": frame, "emotion": "neutral", "confidence": 0.0, "error": str(e)})
        
        # Determine dominant emotion
        emotion_counts = {}
        for e in emotions:
            em = e.get("emotion", "neutral")
            emotion_counts[em] = emotion_counts.get(em, 0) + e.get("confidence", 1)
        
        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        
        return {
            "emotions": emotions,
            "dominant_emotion": dominant,
            "summary": f"Analyzed {len(images)} frames using DeepFace. Dominant emotion: {dominant}"
        }
    
    def _mock_process(
        self, 
        employee_id: str, 
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Mock processing for testing"""
        logger.debug(f"Mock emotion analysis for employee {employee_id}")
        
        emotions = []
        for i, img_data in enumerate(images):
            frame = img_data.get("frame", i)
            # Random mock emotion
            emotion = np.random.choice(self.EMOTIONS)
            confidence = np.random.uniform(0.6, 0.95)
            
            emotions.append({
                "frame": frame,
                "emotion": emotion,
                "confidence": round(confidence, 3)
            })
        
        # Determine dominant
        emotion_counts = {}
        for e in emotions:
            em = e["emotion"]
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
        
        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        
        return {
            "emotions": emotions,
            "dominant_emotion": dominant,
            "summary": f"Mock analysis of {len(images)} frames. Dominant emotion: {dominant}"
        }
