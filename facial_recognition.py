# facial_recognition.py
"""
Comprehensive Facial Recognition System
Author: Arjun Joshi
Date: 12.8.2025

A multipurpose facial recognition system that can:
- Run standalone via CLI
- Be imported as a module into other scripts
- Launch an optional Gradio web interface (--interface flag)
- Process video streams via NVIDIA DeepStream (Jetson optimized)

Features:
- Face enrollment with embedding vectorization
- Face matching against enrollment database
- Live camera streaming with multi-frame averaging
- DeepFace.stream() real-time analysis
- Manual image verification
- Direct DeepFace.verify() comparison
- Facial attribute analysis (age, gender, race, emotion)
- DeepStream integration for hardware-accelerated video processing
  - RTSP stream support (IP cameras, NVRs)
  - USB camera support (V4L2)
  - CSI camera support (Jetson native)
  - Video file processing
  - Multi-source/multi-camera support

Usage:
    # CLI enrollment
    python facial_recognition.py enroll -n "John Doe" -i photo1.jpg photo2.jpg
    
    # CLI matching
    python facial_recognition.py match -t target.jpg
    
    # Launch Gradio interface
    python facial_recognition.py --interface
    
    # DeepStream - RTSP stream
    python facial_recognition.py deepstream -s "rtsp://192.168.1.100:554/stream"
    
    # DeepStream - USB camera
    python facial_recognition.py deepstream -s /dev/video0
    
    # DeepStream - CSI camera (Jetson)
    python facial_recognition.py deepstream -s csi
    
    # DeepStream - Video file
    python facial_recognition.py deepstream -s video.mp4
    
    # DeepStream - Multiple sources
    python facial_recognition.py deepstream-multi -s "rtsp://cam1/stream" "rtsp://cam2/stream"
    
    # Import as module
    from facial_recognition import FaceAuthSystem, DeepStreamProcessor
    system = FaceAuthSystem()
    processor = DeepStreamProcessor(system)
    processor.start("rtsp://camera/stream")
"""

import os

# Configure TensorFlow memory growth BEFORE importing TF/DeepFace
# This prevents TensorFlow from grabbing all GPU memory at once
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import cv2
import numpy as np
import time
import threading
import queue
import logging
import argparse
import pickle
import datetime
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any

# Conditional imports for optional features
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠ DeepFace not installed. Install with: pip install deepface")

try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# DeepStream imports (Jetson/NVIDIA only)
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    import pyds
    DEEPSTREAM_AVAILABLE = True
    
    # GstRtspServer is optional (only needed for RTSP output server, not input)
    try:
        gi.require_version('GstRtspServer', '1.0')
        from gi.repository import GstRtspServer
        RTSP_SERVER_AVAILABLE = True
    except (ValueError, ImportError):
        GstRtspServer = None
        RTSP_SERVER_AVAILABLE = False
        
except (ImportError, ValueError):
    DEEPSTREAM_AVAILABLE = False
    RTSP_SERVER_AVAILABLE = False
    Gst = None
    GLib = None
    GstRtspServer = None
    pyds = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CORE SYSTEM CLASSES
# =============================================================================

class FaceAuthSystem:
    """
    Core facial authentication system handling enrollment, matching, and verification.
    Can be used standalone or as part of a larger application.
    """
    
    SUPPORTED_MODELS = [
        "VGG-Face", "Facenet", "Facenet512", "OpenFace", 
        "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"
    ]
    
    SUPPORTED_DETECTORS = [
        "opencv", "ssd", "dlib", "mtcnn", "fastmtcnn",
        "retinaface", "mediapipe", "yolov8", "yunet", "centerface"
    ]
    
    def __init__(
        self, 
        db_folder: str = "enrolled_faces", 
        model: str = "Facenet512",
        detector: str = "retinaface",
        threshold: float = 0.30
    ):
        """
        Initialize the Face Authentication System.
        
        Args:
            db_folder: Directory to store enrolled face embeddings
            model: Recognition model to use (default: Facenet512)
            detector: Face detector backend (default: retinaface)
            threshold: Cosine distance threshold for matching (lower = stricter)
        """
        self.db_folder = db_folder
        self.model_name = model
        self.detector = detector
        self.threshold = threshold
        self._database_cache = None
        self._cache_timestamp = None
        
        # Ensure DB folder exists
        os.makedirs(self.db_folder, exist_ok=True)
        logger.info(f"FaceAuthSystem initialized with model={model}, detector={detector}")

    def enroll(self, name: str, image_paths: List[str]) -> Optional[np.ndarray]:
        """
        Enroll a person by generating an averaged embedding from multiple images.
        
        Args:
            name: Identifier for the person
            image_paths: List of paths to face images
            
        Returns:
            Master embedding vector or None if enrollment failed
        """
        if not DEEPFACE_AVAILABLE:
            logger.error("DeepFace not available")
            return None
            
        name = name.strip().replace(" ", "_")
        logger.info(f"--- Enrolling {name} ---")
        embeddings = []
        processed_count = 0
        
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue
                
            try:
                result = DeepFace.represent(
                    img_path=path,
                    model_name=self.model_name,
                    detector_backend=self.detector,
                    enforce_detection=True,
                    align=True
                )
                embeddings.append(result[0]["embedding"])
                processed_count += 1
                logger.info(f"✔ Processed: {path}")
            except Exception as e:
                logger.warning(f"⚠ Skipping {path}: {e}")

        if not embeddings:
            logger.error("❌ Failure: No valid faces processed.")
            return None

        # Create 'Master' Vector (Centroid)
        master_vector = np.mean(embeddings, axis=0)
        
        # Save
        save_path = os.path.join(self.db_folder, f"{name}_deepface.pkl")
        data = {
            "name": name,
            "model": self.model_name,
            "embedding": master_vector,
            "enrolled_at": datetime.datetime.now().isoformat(),
            "image_count": processed_count
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        # Invalidate cache
        self._database_cache = None
        
        logger.info(f"✅ Success: Saved {name} to {save_path}")
        return master_vector

    def enroll_from_frames(self, name: str, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Enroll a person from numpy array frames (e.g., from camera).
        
        Args:
            name: Identifier for the person
            frames: List of BGR numpy arrays
            
        Returns:
            Master embedding vector or None if enrollment failed
        """
        if not DEEPFACE_AVAILABLE:
            logger.error("DeepFace not available")
            return None
            
        name = name.strip().replace(" ", "_")
        logger.info(f"--- Enrolling {name} from {len(frames)} frames ---")
        embeddings = []
        
        for i, frame in enumerate(frames):
            try:
                result = DeepFace.represent(
                    img_path=frame,
                    model_name=self.model_name,
                    detector_backend=self.detector,
                    enforce_detection=True,
                    align=True
                )
                embeddings.append(result[0]["embedding"])
                logger.info(f"✔ Processed frame {i+1}/{len(frames)}")
            except Exception as e:
                logger.warning(f"⚠ Skipping frame {i+1}: {e}")

        if not embeddings:
            logger.error("❌ Failure: No valid faces in frames.")
            return None

        master_vector = np.mean(embeddings, axis=0)
        
        save_path = os.path.join(self.db_folder, f"{name}_deepface.pkl")
        data = {
            "name": name,
            "model": self.model_name,
            "embedding": master_vector,
            "enrolled_at": datetime.datetime.now().isoformat(),
            "image_count": len(embeddings)
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        self._database_cache = None
        logger.info(f"✅ Success: Enrolled {name}")
        return master_vector

    def load_database(self, force_reload: bool = False) -> Dict[str, np.ndarray]:
        """
        Load all enrolled face embeddings into memory.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary mapping names to embedding vectors
        """
        # Use cache if available and recent
        if not force_reload and self._database_cache is not None:
            return self._database_cache
            
        db = {}
        if not os.path.exists(self.db_folder):
            return db
            
        for fname in os.listdir(self.db_folder):
            if fname.endswith("_deepface.pkl"):
                path = os.path.join(self.db_folder, fname)
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                        if data.get("model") == self.model_name:
                            db[data["name"]] = data["embedding"]
                except Exception as e:
                    logger.error(f"Error loading {fname}: {e}")
        
        self._database_cache = db
        self._cache_timestamp = datetime.datetime.now()
        return db

    def match(self, target_image_path: str) -> Tuple[str, float, bool]:
        """
        Match a target image against the enrollment database.
        
        Args:
            target_image_path: Path to the image to identify
            
        Returns:
            Tuple of (matched_name, distance_score, is_match)
        """
        if not DEEPFACE_AVAILABLE or not SCIPY_AVAILABLE:
            return "Error", 1.0, False
            
        database = self.load_database()
        if not database:
            logger.warning("Database is empty")
            return "Unknown", 1.0, False

        try:
            result = DeepFace.represent(
                img_path=target_image_path,
                model_name=self.model_name,
                detector_backend=self.detector,
                enforce_detection=True,
                align=True
            )
            target_vector = result[0]["embedding"]
        except Exception as e:
            logger.error(f"Error processing target: {e}")
            return "Error", 1.0, False

        return self._find_best_match(target_vector, database)

    def match_from_frame(self, frame: np.ndarray) -> Tuple[str, float, bool]:
        """
        Match a frame against the enrollment database.
        
        Args:
            frame: BGR numpy array
            
        Returns:
            Tuple of (matched_name, distance_score, is_match)
        """
        if not DEEPFACE_AVAILABLE or not SCIPY_AVAILABLE:
            return "Error", 1.0, False
            
        database = self.load_database()
        if not database:
            return "Unknown", 1.0, False

        try:
            result = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                detector_backend=self.detector,
                enforce_detection=True,
                align=True
            )
            target_vector = result[0]["embedding"]
        except Exception as e:
            logger.debug(f"No face detected in frame: {e}")
            return "No Face", 1.0, False

        return self._find_best_match(target_vector, database)

    def match_averaged(self, frames: List[np.ndarray]) -> Tuple[str, float, bool]:
        """
        Match using averaged embedding from multiple frames.
        
        Args:
            frames: List of BGR numpy arrays
            
        Returns:
            Tuple of (matched_name, distance_score, is_match)
        """
        if not DEEPFACE_AVAILABLE or not SCIPY_AVAILABLE:
            return "Error", 1.0, False
            
        database = self.load_database()
        if not database:
            return "Unknown", 1.0, False

        embeddings = []
        for frame in frames:
            try:
                result = DeepFace.represent(
                    img_path=frame,
                    model_name=self.model_name,
                    detector_backend=self.detector,
                    enforce_detection=True,
                    align=True
                )
                embeddings.append(result[0]["embedding"])
            except Exception:
                continue

        if not embeddings:
            return "No Face", 1.0, False

        # Average the embeddings
        avg_vector = np.mean(embeddings, axis=0)
        return self._find_best_match(avg_vector, database)

    def _find_best_match(
        self, 
        target_vector: np.ndarray, 
        database: Dict[str, np.ndarray]
    ) -> Tuple[str, float, bool]:
        """Find the best matching face in the database."""
        best_match = "Unknown"
        best_score = 1.0

        for name, db_vector in database.items():
            score = cosine(target_vector, db_vector)
            if score < best_score:
                best_score = score
                best_match = name

        is_match = best_score <= self.threshold
        return best_match, best_score, is_match

    def verify_images(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """
        Direct face verification between two images using DeepFace.verify().
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Verification result dictionary
        """
        if not DEEPFACE_AVAILABLE:
            return {"error": "DeepFace not available"}
            
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                detector_backend=self.detector
            )
            return result
        except Exception as e:
            return {"error": str(e), "verified": False}

    def analyze_face(self, img_path: str) -> Dict[str, Any]:
        """
        Analyze facial attributes (age, gender, race, emotion).
        
        Args:
            img_path: Path to image or numpy array
            
        Returns:
            Analysis results dictionary
        """
        if not DEEPFACE_AVAILABLE:
            return {"error": "DeepFace not available"}
            
        try:
            results = DeepFace.analyze(
                img_path=img_path,
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend=self.detector,
                enforce_detection=True
            )
            return results[0] if isinstance(results, list) else results
        except Exception as e:
            return {"error": str(e)}

    def list_enrolled(self) -> List[Dict[str, Any]]:
        """List all enrolled faces with metadata."""
        enrolled = []
        if not os.path.exists(self.db_folder):
            return enrolled
            
        for fname in os.listdir(self.db_folder):
            if fname.endswith("_deepface.pkl"):
                path = os.path.join(self.db_folder, fname)
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                        enrolled.append({
                            "name": data.get("name", "Unknown"),
                            "model": data.get("model", "Unknown"),
                            "enrolled_at": data.get("enrolled_at", "Unknown"),
                            "image_count": data.get("image_count", 0),
                            "file": fname
                        })
                except Exception as e:
                    logger.error(f"Error reading {fname}: {e}")
        return enrolled

    def delete_enrolled(self, name: str) -> bool:
        """Delete an enrolled face."""
        name = name.strip().replace(" ", "_")
        path = os.path.join(self.db_folder, f"{name}_deepface.pkl")
        if os.path.exists(path):
            os.remove(path)
            self._database_cache = None
            return True
        return False


class DatabaseManager:
    """Manages loading images from directory structures."""
    
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def load_images_from_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Transform a directory of IDs into a DataFrame of images.
        
        Expected structure:
            directory_path/
                person_id_1/
                    image1.jpg
                person_id_2/
                    photo.jpg
        
        Returns:
            DataFrame with columns: id, image_path, image, filename, timestamp
        """
        records = []
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return pd.DataFrame(columns=['id', 'image_path', 'image', 'filename', 'timestamp'])
        
        for person_id in os.listdir(directory_path):
            person_path = os.path.join(directory_path, person_id)
            
            if not os.path.isdir(person_path):
                continue
            
            for filename in os.listdir(person_path):
                if not filename.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    continue
                
                image_path = os.path.join(person_path, filename)
                
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(image_path))
                    
                    records.append({
                        'id': person_id,
                        'image_path': image_path,
                        'image': image,
                        'filename': filename,
                        'timestamp': timestamp
                    })
                except Exception as e:
                    logger.error(f"Error loading {image_path}: {e}")
        
        df = pd.DataFrame(records)
        if len(df) > 0:
            logger.info(f"Loaded {len(df)} images from {df['id'].nunique()} unique IDs")
        return df


class CameraManager:
    """Manages camera capture with threading support."""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None

    def start(self) -> bool:
        """Start camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Camera started")
        return True

    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def capture_single(self) -> Optional[np.ndarray]:
        """Capture a single frame (for one-shot use)."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def capture_multiple(self, count: int = 5, interval: float = 2.0) -> List[np.ndarray]:
        """
        Capture multiple frames over time.
        
        Args:
            count: Number of frames to capture
            interval: Seconds between captures
            
        Returns:
            List of captured frames
        """
        frames = []
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return frames
            
        for i in range(count):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                logger.info(f"Captured frame {i+1}/{count}")
            if i < count - 1:
                time.sleep(interval)
        
        cap.release()
        return frames

    def _capture_loop(self):
        """Background capture loop."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
            time.sleep(0.033)  # ~30 FPS


# =============================================================================
# DEEPSTREAM PROCESSOR
# =============================================================================

class DeepStreamProcessor:
    """
    DeepStream-based video processor for efficient hardware-accelerated
    video decoding and frame extraction on NVIDIA Jetson devices.
    
    Supports:
    - RTSP streams (IP cameras, NVRs)
    - USB cameras (V4L2)
    - Video files (MP4, AVI, etc.)
    - CSI cameras (Jetson native)
    
    Usage:
        processor = DeepStreamProcessor(face_system)
        processor.start("rtsp://192.168.1.100:554/stream")
        # or
        processor.start("/dev/video0")  # USB camera
        # or  
        processor.start("video.mp4")  # File
    """
    
    # Source type detection
    SOURCE_TYPE_RTSP = "rtsp"
    SOURCE_TYPE_USB = "usb"
    SOURCE_TYPE_FILE = "file"
    SOURCE_TYPE_CSI = "csi"
    SOURCE_TYPE_BAYER = "bayer"  # For Bayer/RAW cameras like RG10
    
    def __init__(
        self,
        face_system: 'FaceAuthSystem',
        recognition_interval: float = 1.0,
        display_output: bool = True,
        output_width: int = 1280,
        output_height: int = 720
    ):
        """
        Initialize DeepStream processor.
        
        Args:
            face_system: FaceAuthSystem instance for face matching
            recognition_interval: Seconds between face recognition attempts
            display_output: Whether to display video output window
            output_width: Output display width
            output_height: Output display height
        """
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError(
                "DeepStream not available. Ensure you have:\n"
                "  1. NVIDIA DeepStream SDK installed\n"
                "  2. PyDS bindings installed\n"
                "  3. GStreamer with GI bindings\n"
                "Install on Jetson: sudo apt install deepstream-6.* python3-gi python3-gst-1.0"
            )
        
        self.face_system = face_system
        self.recognition_interval = recognition_interval
        self.display_output = display_output
        self.output_width = output_width
        self.output_height = output_height
        
        # Pipeline state
        self.pipeline = None
        self.loop = None
        self.running = False
        self.last_recognition_time = 0
        
        # Results callback
        self._on_recognition_callback = None
        
        # Frame buffer for recognition
        self._current_frame = None
        self._frame_lock = threading.Lock()
        
        # Initialize GStreamer
        Gst.init(None)
        logger.info("DeepStreamProcessor initialized")
    
    def set_recognition_callback(self, callback):
        """
        Set callback function for recognition results.
        
        Callback signature: callback(name: str, distance: float, is_match: bool, frame: np.ndarray)
        """
        self._on_recognition_callback = callback
    
    def _detect_source_type(self, source: str) -> str:
        """Detect the type of video source."""
        if source.startswith("rtsp://") or source.startswith("rtspt://"):
            return self.SOURCE_TYPE_RTSP
        elif source.startswith("bayer://") or source == "bayer":
            # Explicit Bayer camera mode: bayer:///dev/video2 or just "bayer"
            return self.SOURCE_TYPE_BAYER
        elif source.startswith("/dev/video"):
            return self.SOURCE_TYPE_USB
        elif source == "csi" or source.startswith("csi://"):
            return self.SOURCE_TYPE_CSI
        elif os.path.isfile(source):
            return self.SOURCE_TYPE_FILE
        else:
            # Try as USB camera index
            try:
                int(source)
                return self.SOURCE_TYPE_USB
            except ValueError:
                return self.SOURCE_TYPE_FILE
    
    def _create_pipeline(self, source: str):
        """Create GStreamer pipeline based on source type."""
        source_type = self._detect_source_type(source)
        logger.info(f"Creating pipeline for source type: {source_type}")
        
        pipeline = Gst.Pipeline.new("facial-recognition-pipeline")
        
        if source_type == self.SOURCE_TYPE_RTSP:
            pipeline = self._create_rtsp_pipeline(source)
        elif source_type == self.SOURCE_TYPE_USB:
            pipeline = self._create_usb_pipeline(source)
        elif source_type == self.SOURCE_TYPE_CSI:
            pipeline = self._create_csi_pipeline()
        elif source_type == self.SOURCE_TYPE_BAYER:
            pipeline = self._create_bayer_pipeline(source)
        elif source_type == self.SOURCE_TYPE_FILE:
            pipeline = self._create_file_pipeline(source)
        
        return pipeline
    
    def _create_rtsp_pipeline(self, rtsp_uri: str):
        """Create pipeline for RTSP stream."""
        pipeline_str = f"""
            rtspsrc location={rtsp_uri} latency=100 ! 
            rtph264depay ! h264parse ! 
            nvv4l2decoder ! 
            nvvideoconvert ! 
            video/x-raw(memory:NVMM),format=RGBA ! 
            nvvideoconvert ! 
            video/x-raw,format=BGRx ! 
            videoconvert ! 
            video/x-raw,format=BGR ! 
            appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        """
        
        if self.display_output:
            pipeline_str = f"""
                rtspsrc location={rtsp_uri} latency=100 ! 
                rtph264depay ! h264parse ! 
                nvv4l2decoder ! 
                nvvideoconvert ! 
                video/x-raw(memory:NVMM),format=RGBA ! 
                tee name=t
                t. ! queue ! nvvideoconvert ! 
                    video/x-raw,format=BGRx ! videoconvert ! 
                    video/x-raw,format=BGR ! 
                    appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
                t. ! queue ! nv3dsink sync=false
            """
        
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    
    def _create_usb_pipeline(self, device: str):
        """Create pipeline for USB camera (including Bayer/RAW formats)."""
        # Handle both /dev/videoX and integer index
        if device.startswith("/dev/video"):
            device_path = device
        else:
            try:
                idx = int(device)
                device_path = f"/dev/video{idx}"
            except ValueError:
                device_path = "/dev/video0"
        
        # Simple pipeline using CPU conversion (more compatible, avoids argus issues)
        # Works with most USB cameras and V4L2 devices
        pipeline_str = f"""
            v4l2src device={device_path} do-timestamp=true ! 
            video/x-raw ! 
            videoconvert ! 
            videoscale ! 
            video/x-raw,format=BGR,width=640,height=480 ! 
            appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        """
        
        if self.display_output:
            pipeline_str = f"""
                v4l2src device={device_path} do-timestamp=true ! 
                video/x-raw ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,width=640,height=480 ! 
                tee name=t
                t. ! queue ! videoconvert ! video/x-raw,format=BGR ! 
                    appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
                t. ! queue ! videoconvert ! autovideosink sync=false
            """
        
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    
    def _create_bayer_pipeline(self, source: str):
        """Create pipeline for Bayer/RAW cameras (RG10, RG12, etc.) using software debayer."""
        # Extract device path from bayer:///dev/videoX format
        if source.startswith("bayer://"):
            device_path = source[8:]  # Remove "bayer://"
        elif source == "bayer":
            device_path = "/dev/video2"  # Default
        else:
            device_path = "/dev/video2"
        
        # Use bayer2rgb for software debayering (works without argus daemon)
        # bayer2rgb converts Bayer pattern to RGB
        pipeline_str = f"""
            v4l2src device={device_path} do-timestamp=true ! 
            video/x-raw,format=RG10 ! 
            bayer2rgb ! 
            videoconvert ! 
            videoscale ! 
            video/x-raw,format=BGR,width=640,height=480 ! 
            appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        """
        
        if self.display_output:
            pipeline_str = f"""
                v4l2src device={device_path} do-timestamp=true ! 
                video/x-raw,format=RG10 ! 
                bayer2rgb ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,width=640,height=480 ! 
                tee name=t
                t. ! queue ! videoconvert ! video/x-raw,format=BGR ! 
                    appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
                t. ! queue ! videoconvert ! autovideosink sync=false
            """
        
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    
    def _create_csi_pipeline(self):
        """Create pipeline for CSI camera (Jetson native camera)."""
        pipeline_str = f"""
            nvarguscamerasrc ! 
            video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! 
            nvvideoconvert ! 
            video/x-raw,format=BGRx ! 
            videoconvert ! 
            video/x-raw,format=BGR ! 
            appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        """
        
        if self.display_output:
            pipeline_str = f"""
                nvarguscamerasrc ! 
                video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! 
                tee name=t
                t. ! queue ! nvvideoconvert ! video/x-raw,format=BGRx ! 
                    videoconvert ! video/x-raw,format=BGR ! 
                    appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
                t. ! queue ! nv3dsink sync=false
            """
        
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    
    def _create_file_pipeline(self, file_path: str):
        """Create pipeline for video file."""
        # Detect file extension for appropriate demuxer
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.mp4', '.mov', '.m4v']:
            demux = "qtdemux"
        elif ext in ['.avi']:
            demux = "avidemux"
        elif ext in ['.mkv', '.webm']:
            demux = "matroskademux"
        else:
            demux = "decodebin"
        
        if demux == "decodebin":
            pipeline_str = f"""
                filesrc location={file_path} ! 
                decodebin ! 
                videoconvert ! 
                video/x-raw,format=BGR ! 
                appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
            """
        else:
            pipeline_str = f"""
                filesrc location={file_path} ! 
                {demux} ! h264parse ! 
                nvv4l2decoder ! 
                nvvideoconvert ! 
                video/x-raw,format=BGRx ! 
                videoconvert ! 
                video/x-raw,format=BGR ! 
                appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
            """
        
        if self.display_output:
            if demux == "decodebin":
                pipeline_str = f"""
                    filesrc location={file_path} ! 
                    decodebin ! 
                    tee name=t
                    t. ! queue ! videoconvert ! video/x-raw,format=BGR ! 
                        appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
                    t. ! queue ! videoconvert ! xvimagesink sync=false
                """
            else:
                pipeline_str = f"""
                    filesrc location={file_path} ! 
                    {demux} ! h264parse ! 
                    nvv4l2decoder ! 
                    nvvideoconvert ! 
                    video/x-raw(memory:NVMM),format=RGBA ! 
                    tee name=t
                    t. ! queue ! nvvideoconvert ! video/x-raw,format=BGRx ! 
                        videoconvert ! video/x-raw,format=BGR ! 
                        appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
                    t. ! queue ! nv3dsink sync=false
                """
        
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    
    def _on_new_sample(self, sink):
        """Callback when new frame is available from appsink."""
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # Extract frame dimensions
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")
        
        # Map buffer to numpy array
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        try:
            # Convert to numpy array (BGR format)
            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()
            
            with self._frame_lock:
                self._current_frame = frame
            
            # Check if it's time for recognition
            current_time = time.time()
            if current_time - self.last_recognition_time >= self.recognition_interval:
                self.last_recognition_time = current_time
                self._perform_recognition(frame)
                
        finally:
            buf.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _perform_recognition(self, frame: np.ndarray):
        """Perform face recognition on the frame."""
        try:
            name, distance, is_match = self.face_system.match_from_frame(frame)
            
            if name != "No Face":
                if is_match:
                    logger.info(f"✅ MATCH: {name} (distance: {distance:.4f})")
                else:
                    logger.info(f"⛔ Unknown (best guess: {name}, distance: {distance:.4f})")
            
            # Call callback if set
            if self._on_recognition_callback:
                self._on_recognition_callback(name, distance, is_match, frame)
                
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.EOS:
            logger.info("End of stream reached")
            self.stop()
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer error: {err.message}")
            logger.debug(f"Debug info: {debug}")
            self.stop()
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"GStreamer warning: {warn.message}")
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                logger.debug(f"Pipeline state: {old.value_nick} -> {new.value_nick}")
        
        return True
    
    def start(self, source: str, blocking: bool = True):
        """
        Start processing video from the given source.
        
        Args:
            source: Video source (RTSP URL, device path, file path, or 'csi')
            blocking: If True, blocks until stopped. If False, runs in background.
        """
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        logger.info(f"Starting DeepStream pipeline with source: {source}")
        
        try:
            self.pipeline = self._create_pipeline(source)
        except GLib.Error as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise RuntimeError(f"Pipeline creation failed: {e}")
        
        # Get appsink and connect signal
        sink = self.pipeline.get_by_name("sink")
        if sink:
            sink.connect("new-sample", self._on_new_sample)
        else:
            logger.warning("Could not find appsink element")
        
        # Set up bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to start pipeline")
            raise RuntimeError("Failed to start GStreamer pipeline")
        
        self.running = True
        logger.info("Pipeline started successfully")
        
        if blocking:
            self._run_main_loop()
        else:
            self.loop = GLib.MainLoop()
            thread = threading.Thread(target=self._run_main_loop, daemon=True)
            thread.start()
    
    def _run_main_loop(self):
        """Run the GLib main loop."""
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the pipeline."""
        if not self.running:
            return
        
        logger.info("Stopping pipeline...")
        self.running = False
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        
        if self.loop and self.loop.is_running():
            self.loop.quit()
        
        logger.info("Pipeline stopped")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None


class DeepStreamMultiSource:
    """
    Multi-source DeepStream processor for handling multiple video streams.
    
    Useful for multi-camera setups like security systems.
    
    Usage:
        multi = DeepStreamMultiSource(face_system)
        multi.add_source("rtsp://cam1/stream", "Camera 1")
        multi.add_source("rtsp://cam2/stream", "Camera 2")
        multi.start()
    """
    
    def __init__(
        self,
        face_system: 'FaceAuthSystem',
        recognition_interval: float = 1.0
    ):
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError("DeepStream not available")
        
        self.face_system = face_system
        self.recognition_interval = recognition_interval
        self.sources = {}
        self.pipeline = None
        self.loop = None
        self.running = False
        self._frame_buffers = {}
        self._frame_lock = threading.Lock()
        self.last_recognition_time = {}
        
        Gst.init(None)
        logger.info("DeepStreamMultiSource initialized")
    
    def add_source(self, uri: str, name: str = None):
        """Add a video source."""
        if name is None:
            name = f"source_{len(self.sources)}"
        self.sources[name] = uri
        self._frame_buffers[name] = None
        self.last_recognition_time[name] = 0
        logger.info(f"Added source '{name}': {uri}")
    
    def _create_multi_source_pipeline(self):
        """Create a multi-source DeepStream pipeline using nvstreammux."""
        pipeline = Gst.Pipeline.new("multi-source-pipeline")
        
        # Create streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        streammux.set_property("batch-size", len(self.sources))
        streammux.set_property("width", 1280)
        streammux.set_property("height", 720)
        streammux.set_property("batched-push-timeout", 40000)
        streammux.set_property("live-source", 1)
        pipeline.add(streammux)
        
        # Add sources
        for idx, (name, uri) in enumerate(self.sources.items()):
            if uri.startswith("rtsp://"):
                # RTSP source
                source = Gst.ElementFactory.make("rtspsrc", f"source-{idx}")
                source.set_property("location", uri)
                source.set_property("latency", 100)
                
                depay = Gst.ElementFactory.make("rtph264depay", f"depay-{idx}")
                parser = Gst.ElementFactory.make("h264parse", f"parser-{idx}")
                decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{idx}")
                
                pipeline.add(source)
                pipeline.add(depay)
                pipeline.add(parser)
                pipeline.add(decoder)
                
                # Link source to depay (dynamic pad)
                def on_pad_added(src, pad, depay=depay):
                    sink_pad = depay.get_static_pad("sink")
                    if not sink_pad.is_linked():
                        pad.link(sink_pad)
                
                source.connect("pad-added", on_pad_added)
                depay.link(parser)
                parser.link(decoder)
                
                # Link decoder to streammux
                srcpad = decoder.get_static_pad("src")
                sinkpad = streammux.get_request_pad(f"sink_{idx}")
                srcpad.link(sinkpad)
                
            elif uri.startswith("/dev/video"):
                # USB camera
                source = Gst.ElementFactory.make("v4l2src", f"source-{idx}")
                source.set_property("device", uri)
                
                caps = Gst.ElementFactory.make("capsfilter", f"caps-{idx}")
                caps.set_property("caps", Gst.Caps.from_string(
                    "video/x-raw,width=640,height=480,framerate=30/1"
                ))
                
                vidconv = Gst.ElementFactory.make("videoconvert", f"vidconv-{idx}")
                nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv-{idx}")
                
                pipeline.add(source)
                pipeline.add(caps)
                pipeline.add(vidconv)
                pipeline.add(nvvidconv)
                
                source.link(caps)
                caps.link(vidconv)
                vidconv.link(nvvidconv)
                
                srcpad = nvvidconv.get_static_pad("src")
                sinkpad = streammux.get_request_pad(f"sink_{idx}")
                srcpad.link(sinkpad)
        
        # Add converter and tiler for multi-stream display
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
        
        # Calculate tiler dimensions
        num_sources = len(self.sources)
        cols = int(np.ceil(np.sqrt(num_sources)))
        rows = int(np.ceil(num_sources / cols))
        tiler.set_property("rows", rows)
        tiler.set_property("columns", cols)
        tiler.set_property("width", 1280)
        tiler.set_property("height", 720)
        
        pipeline.add(nvvidconv)
        pipeline.add(tiler)
        
        streammux.link(nvvidconv)
        nvvidconv.link(tiler)
        
        # Add display sink
        sink = Gst.ElementFactory.make("nv3dsink", "sink")
        sink.set_property("sync", False)
        pipeline.add(sink)
        tiler.link(sink)
        
        return pipeline
    
    def start(self, blocking: bool = True):
        """Start processing all sources."""
        if not self.sources:
            logger.error("No sources added")
            return
        
        if self.running:
            logger.warning("Already running")
            return
        
        logger.info(f"Starting multi-source pipeline with {len(self.sources)} sources")
        
        self.pipeline = self._create_multi_source_pipeline()
        
        # Set up bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start multi-source pipeline")
        
        self.running = True
        
        if blocking:
            self._run_main_loop()
        else:
            thread = threading.Thread(target=self._run_main_loop, daemon=True)
            thread.start()
    
    def _on_bus_message(self, bus, message):
        """Handle bus messages."""
        if message.type == Gst.MessageType.EOS:
            logger.info("End of stream")
            self.stop()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Error: {err.message}")
            self.stop()
        return True
    
    def _run_main_loop(self):
        """Run main loop."""
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop all processing."""
        if not self.running:
            return
        
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop and self.loop.is_running():
            self.loop.quit()
        logger.info("Multi-source pipeline stopped")


# =============================================================================
# LIVE STREAM RECOGNIZER (Uses pkl database - FAST)
# =============================================================================

class LiveStreamRecognizer:
    """
    Continuous live stream face recognition using pkl embedding database.
    Much faster than DeepFace.stream() because it uses pre-computed embeddings.
    """
    
    def __init__(self, face_system: FaceAuthSystem, camera_index: int = 0):
        self.face_system = face_system
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_result = {"name": "Initializing...", "distance": 0, "is_match": False}
        self.frame_lock = threading.Lock()
        self.last_recognition_time = 0
        self.recognition_interval = 0.5  # Recognize every 0.5 seconds
        
    def start(self):
        """Start the live stream."""
        if self.running:
            return "Already running"
        
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        
        if not self.cap.isOpened():
            return "❌ Failed to open camera"
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return "✅ Live stream started"
    
    def stop(self):
        """Stop the live stream."""
        self.running = False
        if self.cap:
            self.cap.release()
        return "Stream stopped"
    
    def _capture_loop(self):
        """Main capture and recognition loop."""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)  # Mirror
            display_frame = frame.copy()
            
            # Run recognition periodically (not every frame to save CPU)
            current_time = time.time()
            if current_time - self.last_recognition_time >= self.recognition_interval:
                self.last_recognition_time = current_time
                
                try:
                    name, distance, is_match = self.face_system.match_from_frame(frame)
                    self.current_result = {"name": name, "distance": distance, "is_match": is_match}
                except Exception as e:
                    self.current_result = {"name": "Error", "distance": 0, "is_match": False}
            
            # Draw result on frame
            result = self.current_result
            if result["name"] != "No Face" and result["name"] != "Error":
                if result["is_match"]:
                    # Green box and text for match
                    color = (0, 255, 0)
                    text = f"MATCH: {result['name']} ({result['distance']:.3f})"
                else:
                    # Orange for no match
                    color = (0, 165, 255)
                    text = f"Unknown ({result['name']}: {result['distance']:.3f})"
                
                cv2.rectangle(display_frame, (10, 10), (630, 70), color, 2)
                cv2.putText(display_frame, text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                # Gray for no face
                cv2.putText(display_frame, "No face detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
            
            # Convert to RGB for Gradio
            with self.frame_lock:
                self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            time.sleep(0.033)  # ~30 FPS display
    
    def get_frame(self):
        """Get current frame for Gradio display."""
        with self.frame_lock:
            if self.current_frame is not None:
                result = self.current_result
                status = f"{result['name']} | Distance: {result['distance']:.4f} | Match: {result['is_match']}"
                return self.current_frame, status
        return None, "Waiting for camera..."


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

class GradioInterface:
    """
    Gradio web interface for the Facial Recognition System.
    
    Usage:
        system = FaceAuthSystem()
        interface = GradioInterface(system)
        interface.launch(theme=gr.themes.Soft())
    """
    
    def __init__(self, system: FaceAuthSystem):
        self.system = system
        self.demo = None
        
        try:
            import gradio as gr
            self.gr = gr
        except ImportError:
            logger.error("Gradio not installed. Install with: pip install gradio")
            raise ImportError("Gradio is required for the web interface")

    # -------------------------------------------------------------------------
    # Callback Functions
    # -------------------------------------------------------------------------
    def _camera_stream_verify(self):
        """Capture 5 photos over 10 seconds and verify against DB."""
        cam = CameraManager()
        log_messages = ["📸 Starting camera capture..."]
        log_messages.append("⏱ Capturing 5 photos over 10 seconds...")
        
        frames = cam.capture_multiple(count=5, interval=2.0)
        
        if not frames:
            return None, "❌ Failed to capture frames from camera"
        
        log_messages.append(f"✔ Captured {len(frames)} frames")
        
        display_frame = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
        name, distance, is_match = self.system.match_averaged(frames)
        
        if is_match:
            log_messages.append(f"\n✅ MATCH FOUND: {name}")
            log_messages.append(f"📊 Distance: {distance:.4f} (Threshold: {self.system.threshold})")
        else:
            log_messages.append(f"\n⛔ NO MATCH")
            log_messages.append(f"📊 Best guess: {name} (Distance: {distance:.4f})")
        
        return display_frame, "\n".join(log_messages)

    def _start_deepface_stream(self, db_path: str):
        """Start DeepFace real-time stream analysis."""
        if not DEEPFACE_AVAILABLE:
            return "❌ DeepFace not available"
        
        if not db_path or not os.path.exists(db_path):
            return "❌ Please provide a valid database path with face images"
        
        try:
            DeepFace.stream(
                db_path=db_path,
                model_name=self.system.model_name,
                detector_backend=self.system.detector,
                source=0,
                time_threshold=3,
                frame_threshold=3
            )
            return "✅ Stream completed"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _manual_verify(self, images):
        """Verify uploaded images against enrollment database."""
        if not images or len(images) == 0:
            return None, "❌ Please upload at least one image"
        
        log_messages = [f"📸 Processing {len(images)} uploaded image(s)..."]
        frames = []
        display_img = None
        
        for img in images:
            img_path = img if isinstance(img, str) else img.name
            try:
                frame = cv2.imread(img_path)
                if frame is not None:
                    frames.append(frame)
                    if display_img is None:
                        display_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    log_messages.append(f"✔ Loaded: {os.path.basename(img_path)}")
            except Exception:
                log_messages.append(f"⚠ Failed to load: {os.path.basename(img_path)}")
        
        if not frames:
            return None, "\n".join(log_messages) + "\n\n❌ No valid images found"
        
        name, distance, is_match = self.system.match_averaged(frames)
        
        if is_match:
            log_messages.append(f"\n✅ MATCH FOUND: {name}")
            log_messages.append(f"📊 Distance: {distance:.4f} (Threshold: {self.system.threshold})")
        else:
            log_messages.append(f"\n⛔ NO MATCH")
            log_messages.append(f"📊 Best guess: {name} (Distance: {distance:.4f})")
        
        return display_img, "\n".join(log_messages)

    def _enroll_face(self, name: str, images):
        """Enroll a new face into the database."""
        if not name or not name.strip():
            return "❌ Please enter a name"
        
        if not images or len(images) == 0:
            return "❌ Please upload at least one image"
        
        image_paths = []
        for img in images:
            path = img if isinstance(img, str) else img.name
            if os.path.exists(path):
                image_paths.append(path)
        
        if not image_paths:
            return "❌ No valid images found"
        
        result = self.system.enroll(name, image_paths)
        
        if result is not None:
            return f"✅ Successfully enrolled {name}!\n📊 Embedding size: {len(result)}\n🖼 Images processed: {len(image_paths)}"
        else:
            return "❌ Enrollment failed. No valid faces detected."

    def _get_enrolled_list(self):
        """Get list of enrolled faces."""
        enrolled = self.system.list_enrolled()
        if not enrolled:
            return "📋 No faces enrolled yet"
        
        lines = ["📋 Enrolled Faces:\n"]
        for entry in enrolled:
            lines.append(f"• {entry['name']} (Model: {entry['model']}, Images: {entry['image_count']})")
        return "\n".join(lines)

    def _delete_face(self, name: str):
        """Delete an enrolled face."""
        if not name:
            return "❌ Please enter a name to delete"
        
        if self.system.delete_enrolled(name):
            return f"✅ Deleted {name}"
        else:
            return f"❌ {name} not found in database"

    def _direct_verify(self, img1, img2):
        """Direct face verification between two images."""
        if img1 is None or img2 is None:
            return None, None, "❌ Please upload both images"
        
        img1_path = img1 if isinstance(img1, str) else img1.name
        img2_path = img2 if isinstance(img2, str) else img2.name
        
        frame1 = cv2.imread(img1_path)
        frame2 = cv2.imread(img2_path)
        
        display1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) if frame1 is not None else None
        display2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) if frame2 is not None else None
        
        result = self.system.verify_images(img1_path, img2_path)
        
        if "error" in result:
            return display1, display2, f"❌ Error: {result['error']}"
        
        verified = result.get("verified", False)
        distance = result.get("distance", 0)
        threshold = result.get("threshold", 0)
        model = result.get("model", self.system.model_name)
        
        if verified:
            output = f"✅ SAME PERSON\n\n"
        else:
            output = f"⛔ DIFFERENT PEOPLE\n\n"
        
        output += f"📊 Distance: {distance:.4f}\n"
        output += f"📏 Threshold: {threshold:.4f}\n"
        output += f"🤖 Model: {model}"
        
        return display1, display2, output

    def _analyze_attributes(self, image):
        """Analyze facial attributes."""
        if image is None:
            return None, "❌ Please upload an image"
        
        img_path = image if isinstance(image, str) else image.name
        
        frame = cv2.imread(img_path)
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
        
        result = self.system.analyze_face(img_path)
        
        if "error" in result:
            return display, f"❌ Error: {result['error']}"
        
        output_lines = ["🔍 FACIAL ATTRIBUTE ANALYSIS\n"]
        output_lines.append("=" * 40)
        
        age = result.get("age", "Unknown")
        output_lines.append(f"\n👤 Age: {age}")
        
        gender = result.get("dominant_gender", result.get("gender", "Unknown"))
        gender_scores = result.get("gender", {})
        if isinstance(gender_scores, dict):
            output_lines.append(f"\n⚧ Gender: {gender}")
            for g, score in gender_scores.items():
                output_lines.append(f"   • {g}: {score:.1f}%")
        else:
            output_lines.append(f"\n⚧ Gender: {gender}")
        
        race = result.get("dominant_race", "Unknown")
        race_scores = result.get("race", {})
        output_lines.append(f"\n🌍 Race: {race}")
        if isinstance(race_scores, dict):
            sorted_races = sorted(race_scores.items(), key=lambda x: x[1], reverse=True)
            for r, score in sorted_races[:3]:
                output_lines.append(f"   • {r}: {score:.1f}%")
        
        emotion = result.get("dominant_emotion", "Unknown")
        emotion_scores = result.get("emotion", {})
        output_lines.append(f"\n😊 Emotion: {emotion}")
        if isinstance(emotion_scores, dict):
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            for e, score in sorted_emotions[:3]:
                output_lines.append(f"   • {e}: {score:.1f}%")
        
        return display, "\n".join(output_lines)

    # -------------------------------------------------------------------------
    # Build Interface
    # -------------------------------------------------------------------------
    def build(self):
        """Build the Gradio Blocks interface."""
        gr = self.gr
        
        with gr.Blocks() as demo:
            gr.Markdown(f"""
            # 🎯 Comprehensive Facial Recognition System
            
            A multi-purpose facial recognition system powered by [DeepFace](https://github.com/serengil/deepface).
            
            **Current Model:** `{self.system.model_name}` | **Detector:** `{self.system.detector}` | **Threshold:** `{self.system.threshold}`
            """)
            
            # Tab 1: Camera Streaming
            with gr.Tab("📹 Camera Verification"):
                gr.Markdown("""
                ### Live Camera Verification
                Captures 5 photos over 10 seconds and uses averaged embeddings for verification.
                """)
                with gr.Row():
                    with gr.Column():
                        cam_btn = gr.Button("🎥 Start Camera Capture", variant="primary", size="lg")
                    with gr.Column():
                        cam_output_img = gr.Image(label="Captured Frame", type="numpy")
                cam_output_log = gr.Textbox(label="Results", lines=8, interactive=False)
                
                cam_btn.click(fn=self._camera_stream_verify, outputs=[cam_output_img, cam_output_log])
            
            # Tab 1b: Live Stream Recognition (Fast - uses pkl database)
            with gr.Tab("🎬 Live Stream"):
                gr.Markdown("""
                ### ⚡ Continuous Live Recognition
                Real-time face recognition using pre-computed embeddings (pkl database).
                
                **Much faster** than DeepFace.stream() - uses vectorized matching!
                """)
                
                # Initialize live stream recognizer
                live_stream = LiveStreamRecognizer(self.system, camera_index=0)
                
                with gr.Row():
                    live_start_btn = gr.Button("▶️ Start Live Stream", variant="primary", size="lg")
                    live_stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                
                live_status = gr.Textbox(label="Status", value="Click Start to begin", interactive=False)
                live_video = gr.Image(label="Live Feed", height=480)
                live_result = gr.Textbox(label="Recognition Result", interactive=False)
                
                def start_live():
                    return live_stream.start()
                
                def stop_live():
                    return live_stream.stop()
                
                def get_live_frame():
                    frame, status = live_stream.get_frame()
                    return frame, status
                
                live_start_btn.click(fn=start_live, outputs=[live_status])
                live_stop_btn.click(fn=stop_live, outputs=[live_status])
                
                # Timer to poll frames
                live_timer = gr.Timer(value=0.1)  # 10 FPS polling
                live_timer.tick(fn=get_live_frame, outputs=[live_video, live_result])
            
            # Tab 2: Manual Verification
            with gr.Tab("📤 Manual Verification"):
                gr.Markdown("""
                ### Upload Images for Verification
                Upload up to 5 images to verify against the enrollment database using averaged embeddings.
                """)
                with gr.Row():
                    with gr.Column():
                        manual_images = gr.File(
                            label="Upload Face Images (up to 5)",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        manual_btn = gr.Button("🔍 Verify", variant="primary")
                    with gr.Column():
                        manual_output_img = gr.Image(label="Uploaded Image", type="numpy")
                manual_output_log = gr.Textbox(label="Results", lines=10, interactive=False)
                
                manual_btn.click(fn=self._manual_verify, inputs=[manual_images], outputs=[manual_output_img, manual_output_log])
            
            # Tab 3: Enrollment
            with gr.Tab("📝 Enrollment"):
                gr.Markdown("""
                ### Enroll New Face
                Add a new person to the recognition database by uploading their face images.
                """)
                with gr.Row():
                    with gr.Column():
                        enroll_name = gr.Textbox(
                            label="Person's Name",
                            placeholder="Enter name (e.g., John Doe)"
                        )
                        enroll_images = gr.File(
                            label="Upload Face Images",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        enroll_btn = gr.Button("🚀 Enroll Face", variant="primary")
                    with gr.Column():
                        enroll_output = gr.Textbox(label="Enrollment Status", lines=5, interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### Manage Enrolled Faces")
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("🔄 Refresh List")
                        enrolled_list = gr.Textbox(label="Enrolled Faces", lines=8, interactive=False)
                    with gr.Column():
                        delete_name = gr.Textbox(label="Name to Delete", placeholder="Enter name")
                        delete_btn = gr.Button("🗑️ Delete", variant="stop")
                        delete_output = gr.Textbox(label="Delete Status", lines=2, interactive=False)
                
                enroll_btn.click(fn=self._enroll_face, inputs=[enroll_name, enroll_images], outputs=[enroll_output])
                refresh_btn.click(fn=self._get_enrolled_list, outputs=[enrolled_list])
                delete_btn.click(fn=self._delete_face, inputs=[delete_name], outputs=[delete_output])
            
            # Tab 4: Direct Verify
            with gr.Tab("🔗 Direct Verify"):
                gr.Markdown("""
                ### Face Verification (DeepFace.verify)
                Compare two faces directly without using the enrollment database.
                Great for one-off comparisons and demos.
                """)
                with gr.Row():
                    verify_img1 = gr.File(label="First Image", file_types=["image"], type="filepath")
                    verify_img2 = gr.File(label="Second Image", file_types=["image"], type="filepath")
                verify_btn = gr.Button("⚡ Verify Match", variant="primary")
                with gr.Row():
                    verify_display1 = gr.Image(label="Image 1", type="numpy")
                    verify_display2 = gr.Image(label="Image 2", type="numpy")
                verify_output = gr.Textbox(label="Verification Result", lines=6, interactive=False)
                
                verify_btn.click(
                    fn=self._direct_verify,
                    inputs=[verify_img1, verify_img2],
                    outputs=[verify_display1, verify_display2, verify_output]
                )
            
            # Tab 5: Facial Analysis
            with gr.Tab("🔬 Facial Analysis"):
                gr.Markdown("""
                ### Facial Attribute Analysis
                Analyze age, gender, race, and emotion from a face image.
                """)
                with gr.Row():
                    with gr.Column():
                        analysis_img = gr.File(label="Upload Image", file_types=["image"], type="filepath")
                        analysis_btn = gr.Button("🔍 Analyze", variant="primary")
                    with gr.Column():
                        analysis_display = gr.Image(label="Analyzed Image", type="numpy")
                analysis_output = gr.Textbox(label="Analysis Results", lines=15, interactive=False)
                
                analysis_btn.click(
                    fn=self._analyze_attributes,
                    inputs=[analysis_img],
                    outputs=[analysis_display, analysis_output]
                )
            
            # Footer
            gr.Markdown(f"""
            ---
            *Built with [DeepFace](https://github.com/serengil/deepface) • 
            Model: {self.system.model_name} • Detector: {self.system.detector}*
            """)
        
        self.demo = demo
        return demo

    def launch(self, server_port: int = 7860, share: bool = False, theme=None):
        """
        Launch the Gradio interface.
        
        Args:
            server_port: Port to run the server on
            share: Whether to create a public link
            theme: Gradio theme (e.g., gr.themes.Soft())
        """
        if self.demo is None:
            self.build()
        
        self.demo.launch(
            server_port=server_port,
            share=share,
            theme=theme
        )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point for CLI and Gradio interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Facial Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll a new person
  python facial_recognition.py enroll -n "John Doe" -i img1.jpg img2.jpg img3.jpg

  # Match/identify a face
  python facial_recognition.py match -t unknown.jpg

  # Launch Gradio web interface
  python facial_recognition.py --interface

  # Analyze facial attributes
  python facial_recognition.py analyze -i face.jpg
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--interface", "-I",
        action="store_true",
        help="Launch Gradio web interface"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Facenet512",
        choices=FaceAuthSystem.SUPPORTED_MODELS,
        help="Face recognition model (default: Facenet512)"
    )
    parser.add_argument(
        "--detector", "-d",
        type=str,
        default="retinaface",
        choices=FaceAuthSystem.SUPPORTED_DETECTORS,
        help="Face detector backend (default: retinaface)"
    )
    parser.add_argument(
        "--db-folder",
        type=str,
        default="enrolled_faces",
        help="Folder for storing enrolled face embeddings"
    )
    parser.add_argument(
        "--threshold", "-T",
        type=float,
        default=0.30,
        help="Matching threshold (lower = stricter, default: 0.30)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Port for Gradio interface (default: 7860)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enroll subcommand
    parser_enroll = subparsers.add_parser("enroll", help="Enroll a new person")
    parser_enroll.add_argument("-n", "--name", required=True, help="Person's name")
    parser_enroll.add_argument("-i", "--images", nargs="+", required=True, help="Image paths")
    
    # Match subcommand
    parser_match = subparsers.add_parser("match", help="Match/identify a face")
    parser_match.add_argument("-t", "--target", required=True, help="Target image path")
    
    # Verify subcommand
    parser_verify = subparsers.add_parser("verify", help="Verify two faces are same person")
    parser_verify.add_argument("-1", "--img1", required=True, help="First image")
    parser_verify.add_argument("-2", "--img2", required=True, help="Second image")
    
    # Analyze subcommand
    parser_analyze = subparsers.add_parser("analyze", help="Analyze facial attributes")
    parser_analyze.add_argument("-i", "--image", required=True, help="Image to analyze")
    
    # List subcommand
    parser_list = subparsers.add_parser("list", help="List enrolled faces")
    
    # Delete subcommand
    parser_delete = subparsers.add_parser("delete", help="Delete enrolled face")
    parser_delete.add_argument("-n", "--name", required=True, help="Name to delete")
    
    # DeepStream stream subcommand
    parser_ds = subparsers.add_parser("deepstream", help="Start DeepStream video processing")
    parser_ds.add_argument(
        "-s", "--source", 
        required=True,
        help="Video source: RTSP URL (rtsp://...), USB camera (/dev/video0 or index), CSI camera ('csi'), or file path"
    )
    parser_ds.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Recognition interval in seconds (default: 1.0)"
    )
    parser_ds.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display output"
    )
    
    # DeepStream multi-source subcommand
    parser_ds_multi = subparsers.add_parser("deepstream-multi", help="Start multi-source DeepStream processing")
    parser_ds_multi.add_argument(
        "-s", "--sources",
        nargs="+",
        required=True,
        help="Video sources (RTSP URLs or device paths)"
    )
    parser_ds_multi.add_argument(
        "--names",
        nargs="+",
        help="Names for each source (optional)"
    )
    parser_ds_multi.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Recognition interval in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = FaceAuthSystem(
        db_folder=args.db_folder,
        model=args.model,
        detector=args.detector,
        threshold=args.threshold
    )
    
    # Launch Gradio interface
    if args.interface:
        logger.info("Launching Gradio interface...")
        try:
            import gradio as gr
            interface = GradioInterface(system)
            interface.launch(
                server_port=args.port,
                share=False,
                theme=gr.themes.Soft(
                    primary_hue="blue",
                    font=gr.themes.GoogleFont("Inter")
                )
            )
        except ImportError:
            logger.error("Gradio not installed. Install with: pip install gradio")
        return
    
    # Handle CLI commands
    if args.command == "enroll":
        result = system.enroll(args.name, args.images)
        if result is not None:
            print(f"✅ Successfully enrolled {args.name}")
        else:
            print("❌ Enrollment failed")
            
    elif args.command == "match":
        name, distance, is_match = system.match(args.target)
        if is_match:
            print(f"✅ MATCH FOUND: {name}")
            print(f"📊 Distance: {distance:.4f}")
        else:
            print(f"⛔ NO MATCH")
            print(f"📊 Best guess: {name} (Distance: {distance:.4f})")
            
    elif args.command == "verify":
        result = system.verify_images(args.img1, args.img2)
        if result.get("verified"):
            print("✅ SAME PERSON")
        else:
            print("⛔ DIFFERENT PEOPLE")
        if "distance" in result:
            print(f"📊 Distance: {result['distance']:.4f}")
            
    elif args.command == "analyze":
        result = system.analyze_face(args.image)
        if "error" not in result:
            print(f"👤 Age: {result.get('age', 'Unknown')}")
            print(f"⚧ Gender: {result.get('dominant_gender', 'Unknown')}")
            print(f"🌍 Race: {result.get('dominant_race', 'Unknown')}")
            print(f"😊 Emotion: {result.get('dominant_emotion', 'Unknown')}")
        else:
            print(f"❌ Error: {result['error']}")
            
    elif args.command == "list":
        enrolled = system.list_enrolled()
        if enrolled:
            print("📋 Enrolled Faces:")
            for entry in enrolled:
                print(f"  • {entry['name']} (Model: {entry['model']})")
        else:
            print("No faces enrolled")
            
    elif args.command == "delete":
        if system.delete_enrolled(args.name):
            print(f"✅ Deleted {args.name}")
        else:
            print(f"❌ {args.name} not found")
    
    elif args.command == "deepstream":
        if not DEEPSTREAM_AVAILABLE:
            print("❌ DeepStream not available. Ensure you have:")
            print("   1. NVIDIA DeepStream SDK installed")
            print("   2. PyDS bindings installed")
            print("   3. GStreamer with GI bindings")
            print("   Install on Jetson: sudo apt install deepstream-6.* python3-gi python3-gst-1.0")
            return
        
        print(f"🎬 Starting DeepStream processing...")
        print(f"   Source: {args.source}")
        print(f"   Recognition interval: {args.interval}s")
        print(f"   Display: {'Disabled' if args.no_display else 'Enabled'}")
        print(f"   Press Ctrl+C to stop\n")
        
        def on_recognition(name, distance, is_match, frame):
            if name != "No Face":
                if is_match:
                    print(f"✅ MATCH: {name} (distance: {distance:.4f})")
                else:
                    print(f"⛔ Unknown (closest: {name}, distance: {distance:.4f})")
        
        try:
            processor = DeepStreamProcessor(
                face_system=system,
                recognition_interval=args.interval,
                display_output=not args.no_display
            )
            processor.set_recognition_callback(on_recognition)
            processor.start(args.source, blocking=True)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif args.command == "deepstream-multi":
        if not DEEPSTREAM_AVAILABLE:
            print("❌ DeepStream not available")
            return
        
        print(f"🎬 Starting multi-source DeepStream processing...")
        print(f"   Sources: {len(args.sources)}")
        
        try:
            processor = DeepStreamMultiSource(
                face_system=system,
                recognition_interval=args.interval
            )
            
            names = args.names if args.names else [None] * len(args.sources)
            for source, name in zip(args.sources, names):
                processor.add_source(source, name)
                print(f"   • {name or 'auto'}: {source}")
            
            print(f"   Press Ctrl+C to stop\n")
            processor.start(blocking=True)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
