"""
Facial Enrollment System - Simple Elder-Friendly Interface
Uses MediaPipe FaceMesh for automatic head pose detection and guided capture.
Audio notifications guide users through the 5-picture enrollment process.
"""
import logging
import base64
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Configure logging - single configuration point
# Root logger at INFO, our module at DEBUG to reduce noise from other libraries
logging.basicConfig(
    level=logging.INFO,  # Default INFO for all libraries
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Only our module at DEBUG level

# Ensure output is not buffered
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

logger.info("=== Facial Enrollment Logger Initialized (DEBUG level) ===")

# Disable Gradio analytics/telemetry before importing

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import threading
import sqlite3
import json
from deepface import DeepFace

# Import GStreamer camera wrapper for RTSP streams
# This uses the same pipeline as DeepStream recognition
try:
    from gstreamer_camera import GStreamerCamera
    GSTREAMER_AVAILABLE = True
    logger.info("✅ GStreamer camera wrapper available")
except ImportError:
    GSTREAMER_AVAILABLE = False
    logger.warning("⚠️ GStreamer camera wrapper not found, RTSP may have color issues")

global CAMERA_IP, PORT, STREAM, USER, PASSWORD, capture_system, WEBCAM_INDEX
CAMERA_IP = None
PORT = 554
STREAM = "sub"
USER = "admin"
PASSWORD = "Fanatec2025"
capture_system = None
WEBCAM_INDEX = 0
# ============================================================================
# SAFE DATABASE STORAGE (SQLite - no code execution risk unlike pickle)
# ============================================================================

def init_face_database(db_path):
    """Initialize the SQLite database for face embeddings."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            model TEXT NOT NULL,
            detector TEXT,
            embedding BLOB NOT NULL,
            embedding_normalized BLOB,
            image_count INTEGER,
            enrolled_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def save_face_embedding(db_path, name, model, detector, embedding, embedding_normalized, image_count):
    """Save a face embedding to the SQLite database."""
    import datetime
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert numpy arrays to bytes
    embedding_bytes = embedding.astype(np.float32).tobytes()
    normalized_bytes = embedding_normalized.astype(np.float32).tobytes() if embedding_normalized is not None else None
    
    # Insert or replace (update if name exists)
    cursor.execute("""
        INSERT OR REPLACE INTO faces 
        (name, model, detector, embedding, embedding_normalized, image_count, enrolled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        model,
        detector,
        embedding_bytes,
        normalized_bytes,
        image_count,
        datetime.datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    return True


def load_all_face_embeddings(db_path, model_filter=None):
    """Load all face embeddings from the SQLite database."""
    if not os.path.exists(db_path):
        return {}
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if model_filter:
        cursor.execute("SELECT * FROM faces WHERE model = ?", (model_filter,))
    else:
        cursor.execute("SELECT * FROM faces")
    
    faces = {}
    for row in cursor.fetchall():
        embedding = np.frombuffer(row['embedding'], dtype=np.float32)
        faces[row['name']] = {
            'embedding': embedding,
            'model': row['model'],
            'detector': row['detector'],
            'image_count': row['image_count'],
            'enrolled_at': row['enrolled_at']
        }
    
    conn.close()
    return faces


def get_image_base64(image_path):
    """Convert an image file to a base64 data URI for embedding in markdown."""
    if not os.path.exists(image_path):
        return ""
    
    # Determine MIME type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(ext, 'image/png')
    
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{data}"

# Audio support
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ pygame not installed - audio notifications disabled. Install with: pip install pygame")

# Configuration
# Updated to match recognition system for optimal compatibility
CHOSEN_MODEL = "ArcFace"  # Best accuracy (was Facenet512)
# Detector options: retinaface (most accurate), yolov8 (fast + accurate), mtcnn (good balance)
# Using 'yolov8' for modern, fast, and accurate face detection
CHOSEN_DETECTOR = "yolov8"  # Faster and more modern than mtcnn
OUTPUT_DIR = "enrolled_faces"
AUDIO_DIR = "audio"
COUNTDOWN_SECONDS = 3
POSE_HOLD_TIME = 0.5  # How long to hold pose before countdown starts

# Capture targets - what head orientation we want for each photo
# NORMALIZED: pitch 0 = straight, positive = up, negative = down
# yaw 0 = centered, positive = left, negative = right
# WIDE TOLERANCES for easier capture
CAPTURE_TARGETS = [
    {"name": "Front", "yaw_range": (-25, 25), "pitch_range": (-20, 20), 
     "icon": "Front", "instruction": "Look straight at the camera", "audio": "look_forward.mp3"},
    {"name": "Left", "yaw_range": (10, 50), "pitch_range": (-30, 30), 
     "icon": "Left", "instruction": "Turn your head LEFT", "audio": "turn_left.mp3"},
    {"name": "Right", "yaw_range": (-50, -10), "pitch_range": (-30, 30), 
     "icon": "Right", "instruction": "Turn your head RIGHT", "audio": "turn_right.mp3"},
    {"name": "Up", "yaw_range": (-30, 30), "pitch_range": (-50, -15), 
     "icon": "Up", "instruction": "Tilt your chin UP slightly", "audio": "look_up.mp3"},
    {"name": "Down", "yaw_range": (-40, 40), "pitch_range": (15,50 ), 
     "icon": "Down", "instruction": "Tilt your chin DOWN slightly", "audio": "look_down.mp3"},
]

# Secondary guidance audio files - for re-orientation corrections
# These are played when user needs to adjust position during capture
SECONDARY_AUDIO = {
    # Yaw corrections
    "left_exceeded": "guidance_left_exceeded.mp3",      # "Too far left, come back right"
    "left_more": "guidance_left_more.mp3",              # "Turn more to your left"
    "right_exceeded": "guidance_right_exceeded.mp3",    # "Too far right, come back left"
    "right_more": "guidance_right_more.mp3",            # "Turn more to your right"
    # Pitch corrections  
    "up_exceeded": "guidance_up_exceeded.mp3",          # "Too far up, lower chin"
    "up_more": "guidance_up_more.mp3",                  # "Tilt chin up more"
    "down_exceeded": "guidance_down_exceeded.mp3",      # "Too far down, raise chin"
    "down_more": "guidance_down_more.mp3",              # "Tilt chin down more"
    # Secondary axis corrections
    "level_chin": "guidance_level_chin.mp3",            # "Good turn, now level your chin"
    "face_forward": "guidance_face_forward.mp3",        # "Good tilt, now face forward"
}


class AudioPlayer:
    """
    Non-blocking audio player for enrollment cues.
    Uses pygame.mixer for reliable cross-platform audio playback.
    """
    
    def __init__(self, audio_dir=AUDIO_DIR):
        self.audio_dir = audio_dir
        self.enabled = AUDIO_AVAILABLE
        self.last_played = {}  # Track last play time to avoid rapid repeats
        self.min_repeat_interval = 2.0  # Minimum seconds between same audio
        
        # Pre-load audio files for faster playback
        self.sounds = {}
        if self.enabled:
            self._preload_sounds()
    
    def _preload_sounds(self):
        """Pre-load all audio files into memory."""
        # Primary audio files
        audio_files = [
            "beep.mp3",
            "capture_complete.mp3",
            "hold_pose.mp3",
            "look_down.mp3",
            "look_up.mp3",
            "turn_left.mp3",
            "turn_right.mp3",
        ]
        
        # Add secondary guidance audio files
        audio_files.extend(SECONDARY_AUDIO.values())
        
        for filename in audio_files:
            filepath = os.path.join(self.audio_dir, filename)
            if os.path.exists(filepath):
                try:
                    self.sounds[filename] = pygame.mixer.Sound(filepath)
                    logger.debug(f"Loaded audio: {filename}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
            else:
                # Only warn for secondary files (they may not exist yet)
                if filename in SECONDARY_AUDIO.values():
                    logger.debug(f"Secondary audio not found (optional): {filename}")
    
    def play(self, filename, force=False):
        """
        Play an audio file (non-blocking).
        
        Args:
            filename: Name of audio file (e.g., "turn_left.mp3")
            force: If True, play even if recently played
        """
        if not self.enabled or filename is None:
            return
        
        # Check if we should skip due to recent play
        now = time.time()
        if not force and filename in self.last_played:
            if now - self.last_played[filename] < self.min_repeat_interval:
                return
        
        self.last_played[filename] = now
        
        # Play from preloaded sounds if available
        if filename in self.sounds:
            try:
                self.sounds[filename].play()
            except Exception as e:
                print(f"⚠️ Audio playback error: {e}")
        else:
            # Fallback: try to load and play
            filepath = os.path.join(self.audio_dir, filename)
            if os.path.exists(filepath):
                try:
                    sound = pygame.mixer.Sound(filepath)
                    sound.play()
                except Exception as e:
                    print(f"⚠️ Could not play {filename}: {e}")
    
    def play_direction(self, target_name):
        """Play the appropriate direction audio for a capture target."""
        audio_map = {
            "Left": "turn_left.mp3",
            "Right": "turn_right.mp3",
            "Up": "look_up.mp3",
            "Down": "look_down.mp3",
        }
        if target_name in audio_map:
            self.play(audio_map[target_name])
    
    def play_hold(self):
        """Play 'hold pose' audio."""
        self.play("hold_pose.mp3")
    
    def play_beep(self):
        """Play countdown beep."""
        self.play("beep.mp3", force=True)
    
    def play_complete(self):
        """Play capture complete audio."""
        self.play("capture_complete.mp3", force=True)


# Global audio player instance
audio_player = AudioPlayer()


class GuidedEnrollmentCapture:
    """Handles camera capture with MediaPipe face mesh and auto-detection."""
    
    def __init__(self, camera_index=0, camera_ip=None, port = 554, stream = "sub", user = "admin", password = "Fanatec2025"):
        self.camera_index = camera_index
        self.camera_ip = camera_ip
        self.port = port
        self.stream = stream
        self.user = user
        self.password = password
        
        # Construct RTSP URL from components if camera_ip is provided
        if self.camera_ip:
            if stream == "sub":
                self.rtsp_url = f"rtsp://{self.user}:{self.password}@{self.camera_ip}/Preview_01_sub"
            elif stream == "main":
                self.rtsp_url = f"rtsp://{self.user}:{self.password}@{self.camera_ip}:{self.port}"
        else:
            self.rtsp_url = None
        self.cap = None
        self.running = False
        self.camera_ready = False
        self.current_frame = None      # Frame with UI overlays (for display)
        self.raw_frame = None          # Clean frame without overlays (for DeepFace)
        self.frame_lock = threading.Lock()
        
        # MediaPipe Face Mesh - DEFERRED initialization for Jetson performance
        # Models won't load until start_camera() is called
        self.mp_face_mesh = None
        self.face_mesh = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self._mediapipe_initialized = False
        
        # Camera matrix (initialized on first frame)
        self.cam_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # 3D face model points for pose estimation
        self.face_3d = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye corner
            (225.0, 170.0, -135.0),     # Right eye corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ], dtype=np.float64)
        self.face_2d_indices = [1, 152, 33, 263, 61, 291]
        
        # Capture state
        self.user_name = ""
        self.current_step = 0
        self.captured_frames = []
        self.countdown_active = False
        self.countdown_start = 0
        self.pose_held_since = 0
        self.last_pose_valid = False
        self.capture_complete = False
        self.completion_time = 0  # Timestamp when capture completed (for auto-revert)
        
        # "Locked in" state - once countdown starts, we're more forgiving
        self.locked_yaw = 0
        self.locked_pitch = 0
        self.countdown_tolerance = 40  # Extra degrees of tolerance during countdown
        self.enrollment_result = ""
        
        # Current telemetry
        self.current_yaw = 0
        self.current_pitch = 0
        self.face_detected = False
        
        # Audio notification state
        self.last_step_audio_played = -1  # Track which step's direction audio was played
        self.hold_audio_played = False  # Track if hold_pose audio was played for current pose
        self.last_countdown_beep = -1  # Track last countdown number that beeped
        self.completion_audio_played = False  # Track if completion audio was played
        
        # Advisory audio state (rate-limited guidance)
        self.last_advisory_audio_time = 0      # Timestamp of last advisory audio
        self.advisory_audio_count = 0          # Count of audio prompts this capture
        self.advisory_audio_interval = 3.0     # Seconds between audio prompts
        self.advisory_audio_max = 5            # Max prompts per capture (10 seconds total)
        
        # Stability tracking for countdown trigger
        self.stable_frame_count = 0            # Consecutive frames in correct position
        self.stable_frame_threshold = 10       # Frames required before countdown starts (~0.67s at 15fps)

    def _init_mediapipe(self):
        """Initialize MediaPipe models - called lazily to save resources on Jetson."""
        if self._mediapipe_initialized:
            return
        
        logger.info("🔄 Initializing MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self._mediapipe_initialized = True
        logger.info("✅ MediaPipe initialized")

    def start_camera(self):
        """Start camera capture thread."""
        if self.running:
            return

        # Initialize MediaPipe on first camera start (deferred for Jetson performance)
        self._init_mediapipe()
        if self.camera_ip:
            # Use GStreamer wrapper for RTSP to match DeepStream recognition pipeline
            if GSTREAMER_AVAILABLE:
                logger.info(f"Opening RTSP via GStreamer (DeepStream-compatible): {self.rtsp_url}")
                self.cap = GStreamerCamera(self.rtsp_url, width=640, height=480)
                self.cap.open()
            else:
                # Fallback to FFmpeg (may have color issues)
                logger.warning(f"GStreamer wrapper not available, using FFmpeg (color may mismatch DeepStream)")
                logger.info(f"Opening RTSP stream via FFmpeg: {self.rtsp_url}")
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('B','G','R','3'))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 60)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        if not self.cap.isOpened():
            return False
        
        # Warm up camera - discard first few frames
        for _ in range(10):
            self.cap.read()
        
        self.running = True
        self.camera_ready = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_camera(self):
        """Stop camera."""
        self.running = False
        self.camera_ready = False
        if self.cap:
            self.cap.release()

    def reset_capture(self):
        """Reset for new user."""
        self.current_step = 0
        self.captured_frames = []
        self.countdown_active = False
        self.pose_held_since = 0
        self.last_pose_valid = False
        self.capture_complete = False
        self.completion_time = 0
        self.enrollment_result = ""
        
        # Reset audio state
        self.last_step_audio_played = -1
        self.hold_audio_played = False
        self.last_countdown_beep = -1
        self.completion_audio_played = False
        
        # Reset advisory audio state
        self.last_advisory_audio_time = 0
        self.advisory_audio_count = 0
        
        # Reset stability tracking
        self.stable_frame_count = 0

    def set_user_name(self, name):
        """Set the user name for enrollment."""
        self.user_name = name.strip().replace(" ", "_")

    def _put_text_outlined(self, frame, text, pos, font, scale, color, thickness):
        """Draw text with outline for visibility."""
        cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness + 3)
        cv2.putText(frame, text, pos, font, scale, color, thickness)

    def _draw_border(self, frame, color, thickness=2):
        """Draw a border around the frame with black outline for smooth look."""
        h, w = frame.shape[:2]
        margin = 8
        # Outer black outline (drawn first)
        cv2.rectangle(frame, (margin-2, margin-2), (w-margin+2, h-margin+2), (0, 0, 0), 1)
        # Colored border
        cv2.rectangle(frame, (margin, margin), (w-margin, h-margin), color, thickness)
        # Inner black outline
        cv2.rectangle(frame, (margin+thickness+1, margin+thickness+1), 
                     (w-margin-thickness-1, h-margin-thickness-1), (0, 0, 0), 1)

    def _check_pose_match(self, yaw, pitch, target):
        """Check if current pose matches target."""
        yaw_ok = target["yaw_range"][0] <= yaw <= target["yaw_range"][1]
        pitch_ok = target["pitch_range"][0] <= pitch <= target["pitch_range"][1]
        return yaw_ok and pitch_ok

    def _capture_loop(self):
        """Main capture loop with pose detection."""
        frame_count = 0
        process_every_n = 2  # Only process every Nth frame (reduces CPU by ~50%)
        target_fps = 15  # Target processing FPS
        frame_time = 1.0 / target_fps

        while self.running and self.cap.isOpened():
            loop_start = time.time()

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_count += 1

            # Debug: Log first frame to verify color format
            if frame_count == 1:
                logger.info(f"First frame received: shape={frame.shape}, dtype={frame.dtype}, mean={frame.mean():.2f}")
                if hasattr(self.cap, 'getBackendName'):
                    logger.info(f"VideoCapture backend: {self.cap.getBackendName()}")
            
            # Only process every Nth frame to reduce CPU load
            if frame_count % process_every_n == 0:
                # Store raw frame (flipped, no overlays) BEFORE processing
                raw = cv2.flip(frame.copy(), 1)  # Mirror to match display
                
                processed = self._process_frame(frame)
                
                with self.frame_lock:
                    self.raw_frame = raw  # Clean frame for DeepFace
                    self.current_frame = processed  # Frame with overlays for display
            
            # Limit loop speed to target FPS
            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self, frame):
        """Process frame with face mesh and auto-capture logic."""
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction
        
        # Initialize camera matrix
        if self.cam_matrix is None:
            focal_length = w
            self.cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        self.face_detected = False
        pitch, yaw = 0.0, 0.0
        
        if results.multi_face_landmarks:
            self.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw face mesh overlay with thin lines
            # Custom drawing specs for thinner appearance
            thin_tesselation = self.mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1, circle_radius=0)
            thin_contours = self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=0)
            thin_iris = self.mp_drawing.DrawingSpec(color=(48, 255, 255), thickness=1, circle_radius=0)
            
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=thin_tesselation
            )
            
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=thin_contours
            )
            
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=thin_iris
            )
            
            # Calculate head pose
            face_2d = []
            for idx in self.face_2d_indices:
                lm = face_landmarks.landmark[idx]
                face_2d.append([int(lm.x * w), int(lm.y * h)])
            face_2d = np.array(face_2d, dtype=np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(
                self.face_3d, face_2d, self.cam_matrix, self.dist_coeffs
            )
            
            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                
                # Extract Euler angles properly from rotation matrix
                sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
                singular = sy < 1e-6
                
                if not singular:
                    raw_pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
                    yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
                else:
                    raw_pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
                    yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
                
                # Normalize pitch: raw ±180 = straight, make it so 0 = straight
                # When looking straight: raw_pitch ≈ 180 or -180
                # When looking up: raw_pitch goes towards 0 from positive side
                # When looking down: raw_pitch goes towards 0 from negative side
                if raw_pitch > 90:
                    pitch = raw_pitch - 180  # e.g., 180 -> 0, 170 -> -10 (looking down)
                elif raw_pitch < -90:
                    pitch = raw_pitch + 180  # e.g., -180 -> 0, -170 -> 10 (looking up)
                else:
                    # Already in a reasonable range (unlikely for face-forward poses)
                    pitch = raw_pitch
                
                # Now: pitch 0 = straight, positive = up, negative = down
                self.current_yaw = yaw
                self.current_pitch = pitch
        
        # Draw the UI overlay
        frame = self._draw_capture_ui(frame, yaw, pitch)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _draw_capture_ui(self, frame, yaw, pitch):
        """Draw the capture UI with instructions and countdown, with audio cues."""
        h, w = frame.shape[:2]
        
        if self.capture_complete:
            # Play completion audio (once)
            if not self.completion_audio_played:
                audio_player.play_complete()
                self.completion_audio_played = True
            
            # Show completion message
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 100, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            self._put_text_outlined(frame, "ALL DONE!", (w//2 - 120, h//2 - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            self._put_text_outlined(frame, "Processing your enrollment...", (w//2 - 180, h//2 + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
            return frame
        
        if self.current_step >= len(CAPTURE_TARGETS):
            return frame
        
        target = CAPTURE_TARGETS[self.current_step]
        
        # Play direction audio when step changes (once per step)
        if self.current_step != self.last_step_audio_played:
            self.last_step_audio_played = self.current_step
            self.hold_audio_played = False  # Reset hold audio for new step
            self.last_countdown_beep = -1  # Reset countdown beeps
            
            # Play the INITIAL direction audio for this step (force=True to bypass rate limit)
            if target.get("audio"):
                audio_player.play(target["audio"], force=True)
                logger.info(f"🔊 INITIAL audio: {target['audio']} for step {self.current_step + 1} ({target['name']})")
            elif self.current_step == 0:
                # For "Front" - play a beep to indicate start
                audio_player.play_beep()
                logger.info(f"🔊 INITIAL beep for step 1 (Front)")
            
            # Set initial 2-second buffer before secondary guidance can start
            # (by pretending some time already passed, first guidance can play after 2 real seconds)
            initial_buffer = 2.0  # Seconds to wait after initial audio
            self.last_advisory_audio_time = time.time() - (self.advisory_audio_interval - initial_buffer)
            self.advisory_audio_count = 0  # Reset guidance count for new step
        
        # Progress bar at top
        progress_width = int((self.current_step / len(CAPTURE_TARGETS)) * w)
        cv2.rectangle(frame, (0, 0), (w, 8), (50, 50, 50), -1)
        cv2.rectangle(frame, (0, 0), (progress_width, 8), (0, 255, 100), -1)
        
        # Step indicator
        step_text = f"Photo {self.current_step + 1} of {len(CAPTURE_TARGETS)}"
        self._put_text_outlined(frame, step_text, (w//2 - 80, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Main instruction
        self._put_text_outlined(frame, target["instruction"], (w//2 - 200, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Check if pose matches target
        pose_valid = self.face_detected and self._check_pose_match(yaw, pitch, target)
        
        # If countdown is active, use MUCH wider tolerance (locked-in mode)
        if self.countdown_active:
            # Check if still roughly in position (very forgiving during countdown)
            yaw_ok = abs(yaw - self.locked_yaw) < self.countdown_tolerance
            pitch_ok = abs(pitch - self.locked_pitch) < self.countdown_tolerance
            still_locked = self.face_detected and yaw_ok and pitch_ok
            
            elapsed = time.time() - self.countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed
            
            if remaining <= 0:
                # Countdown finished - CAPTURE regardless of slight movement!
                self._do_capture(frame)
                self.countdown_active = False
            elif not still_locked:
                # User moved too much - cancel countdown and reset audio state
                self.countdown_active = False
                self.hold_audio_played = False
                self.last_countdown_beep = -1
                self._put_text_outlined(frame, "Hold still! Try again...", (w//2 - 150, h - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
                self._draw_border(frame, (0, 100, 255), 2)  # Orange border
                
            else:
                # Still locked in - show countdown
                countdown_num = int(remaining) + 1
                
                # Play beep for each countdown number (once per number)
                if countdown_num != self.last_countdown_beep:
                    self.last_countdown_beep = countdown_num
                    audio_player.play_beep()
                
                # Big countdown number
                self._put_text_outlined(frame, str(countdown_num), (w//2 - 40, h//2 + 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
                
                # Progress ring
                progress = (COUNTDOWN_SECONDS - remaining) / COUNTDOWN_SECONDS
                end_angle = int(360 * progress)
                cv2.ellipse(frame, (w//2, h//2), (100, 100), -90, 0, end_angle, (0, 255, 0), 8)
                
                # Green border during countdown
                self._draw_border(frame, (0, 255, 0), 2)
        
        elif pose_valid:
            # Pose is correct - increment stability counter
            self.stable_frame_count += 1
            
            # Reset advisory audio count when pose is achieved
            self.advisory_audio_count = 0
            
            # Check if stable for enough frames before starting countdown
            if self.stable_frame_count >= self.stable_frame_threshold:
                # Stable enough - start building up to countdown
                if not self.last_pose_valid or self.stable_frame_count == self.stable_frame_threshold:
                    self.pose_held_since = time.time()
                
                hold_duration = time.time() - self.pose_held_since
                
                if hold_duration >= POSE_HOLD_TIME:
                    # Start countdown and LOCK IN current pose
                    self.countdown_active = True
                    self.countdown_start = time.time()
                    self.locked_yaw = yaw
                    self.locked_pitch = pitch
                else:
                    # Play hold pose audio (once per pose detection)
                    if not self.hold_audio_played:
                        audio_player.play_hold()
                        self.hold_audio_played = True
                    
                    # Pose correct and stable, waiting to start countdown
                    self._put_text_outlined(frame, "HOLD STILL...", (w//2 - 100, h - 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Green border
                    self._draw_border(frame, (0, 255, 0), 2)
            else:
                # Pose correct but not stable yet - show stabilizing message
                self._put_text_outlined(frame, "Hold steady...", (w//2 - 100, h - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
                # Cyan border (transitioning)
                self._draw_border(frame, (200, 200, 0), 2)
        else:
            # Pose not correct - reset stability and hold audio
            self.stable_frame_count = 0
            self.hold_audio_played = False
            
            # Pose not correct - show guidance
            if not self.face_detected:
                self._put_text_outlined(frame, "No face detected", (w//2 - 120, h - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self._draw_border(frame, (0, 0, 255), 2)  # Red border
            else:
                # Show guidance arrows
                self._draw_guidance(frame, yaw, pitch, target)
                self._draw_border(frame, (0, 165, 255), 2)  # Orange border
        
        self.last_pose_valid = pose_valid
        
        # Thumbnails now displayed in Gradio UI below video (not on frame)
        # self._draw_thumbnails(frame)
        
        return frame

    def _draw_guidance(self, frame, yaw, pitch, target):
        """Draw contextual, magnitude-aware guidance with rate-limited audio."""
        h, w = frame.shape[:2]
        target_name = target["name"]
        
        # Show current pose values for debugging (small text in corner)
        debug_text = f"Yaw:{yaw:.0f} Pitch:{pitch:.0f} Stable:{self.stable_frame_count}"
        self._put_text_outlined(frame, debug_text, (10, h - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        yaw_min, yaw_max = target["yaw_range"]
        pitch_min, pitch_max = target["pitch_range"]
        
        # Calculate deviations
        yaw_low = yaw < yaw_min
        yaw_high = yaw > yaw_max
        pitch_low = pitch < pitch_min
        pitch_high = pitch > pitch_max
        
        yaw_ok = not yaw_low and not yaw_high
        pitch_ok = not pitch_low and not pitch_high
        
        hints = []
        audio_hint = None  # For rate-limited secondary audio
        
        # Context-specific guidance based on target
        if target_name == "Left":
            # Primary axis is yaw (need positive yaw 10-50)
            if yaw_ok:
                if not pitch_ok:
                    hints.append("Good turn! Level your chin")
                    audio_hint = SECONDARY_AUDIO.get("level_chin")
            else:
                if yaw_high:  # Too far left (>50)
                    hints.append("Too far! Come back RIGHT")
                    audio_hint = SECONDARY_AUDIO.get("left_exceeded") or "turn_right.mp3"
                else:  # Not far enough (<10)
                    deviation = yaw_min - yaw
                    if deviation > 15:
                        hints.append("Keep turning LEFT")
                    else:
                        hints.append("Almost! A bit more LEFT")
                    audio_hint = SECONDARY_AUDIO.get("left_more") or "turn_left.mp3"
                    
        elif target_name == "Right":
            # Primary axis is yaw (need negative yaw -50 to -10)
            if yaw_ok:
                if not pitch_ok:
                    hints.append("Good turn! Level your chin")
                    audio_hint = SECONDARY_AUDIO.get("level_chin")
            else:
                if yaw_low:  # Too far right (<-50)
                    hints.append("Too far! Come back LEFT")
                    audio_hint = SECONDARY_AUDIO.get("right_exceeded") or "turn_left.mp3"
                else:  # Not far enough (>-10)
                    deviation = yaw - yaw_max
                    if deviation > 15:
                        hints.append("Keep turning RIGHT")
                    else:
                        hints.append("Almost! A bit more RIGHT")
                    audio_hint = SECONDARY_AUDIO.get("right_more") or "turn_right.mp3"
                    
        elif target_name == "Up":
            # Primary axis is pitch (need negative pitch -50 to -15)
            if pitch_ok:
                if not yaw_ok:
                    hints.append("Good tilt! Face forward")
                    audio_hint = SECONDARY_AUDIO.get("face_forward")
            else:
                if pitch_low:  # Too far up (<-50)
                    hints.append("Too far! Lower chin slightly")
                    audio_hint = SECONDARY_AUDIO.get("up_exceeded") or "look_down.mp3"
                else:  # Not far enough (>-15)
                    deviation = pitch - pitch_max
                    if deviation > 15:
                        hints.append("Tilt chin UP more")
                    else:
                        hints.append("Almost! A bit more UP")
                    audio_hint = SECONDARY_AUDIO.get("up_more") or "look_up.mp3"
                    
        elif target_name == "Down":
            # Primary axis is pitch (need positive pitch 15 to 50)
            if pitch_ok:
                if not yaw_ok:
                    hints.append("Good tilt! Face forward")
                    audio_hint = SECONDARY_AUDIO.get("face_forward")
            else:
                if pitch_high:  # Too far down (>50)
                    hints.append("Too far! Raise chin slightly")
                    audio_hint = SECONDARY_AUDIO.get("down_exceeded") or "look_up.mp3"
                else:  # Not far enough (<15)
                    deviation = pitch_min - pitch
                    if deviation > 15:
                        hints.append("Tilt chin DOWN more")
                    else:
                        hints.append("Almost! A bit more DOWN")
                    audio_hint = SECONDARY_AUDIO.get("down_more") or "look_down.mp3"
                    
        else:
            # Front - use "exceeded" audio variants (they say "come back left/right" etc.)
            if yaw_low:  # Looking too far right, need to turn left
                hints.append("<-- Turn LEFT")
                audio_hint = SECONDARY_AUDIO.get("right_exceeded") or "turn_left.mp3"
            elif yaw_high:  # Looking too far left, need to turn right
                hints.append("Turn RIGHT -->")
                audio_hint = SECONDARY_AUDIO.get("left_exceeded") or "turn_right.mp3"
            # Pitch guidance (only if yaw is ok, to avoid conflicting audio)
            if not audio_hint:
                if pitch_low:  # Looking too far up, need to look down
                    hints.append("Look DOWN v")
                    audio_hint = SECONDARY_AUDIO.get("up_exceeded") or "look_down.mp3"
                elif pitch_high:  # Looking too far down, need to look up
                    hints.append("Look UP ^")
                    audio_hint = SECONDARY_AUDIO.get("down_exceeded") or "look_up.mp3"
            else:
                # Still show pitch hints visually, just don't override audio
                if pitch_low:
                    hints.append("Look DOWN v")
                elif pitch_high:
                    hints.append("Look UP ^")
        
        # Display visual hints
        if hints:
            hint_text = " | ".join(hints)
            text_size = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = (w - text_size[0]) // 2
            self._put_text_outlined(frame, hint_text, (text_x, h - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        
        # Rate-limited audio feedback (2 second interval, max 5 per step)
        if audio_hint and self.advisory_audio_count < self.advisory_audio_max:
            current_time = time.time()
            time_since_last = current_time - self.last_advisory_audio_time
            if time_since_last >= self.advisory_audio_interval:
                audio_player.play(audio_hint)
                self.last_advisory_audio_time = current_time
                self.advisory_audio_count += 1
                logger.info(f"🔊 GUIDANCE audio [{self.advisory_audio_count}/{self.advisory_audio_max}]: {audio_hint}")

    def _draw_checkmark(self, frame, cx, cy, size=10, color=(0, 255, 0), thickness=2):
        """Draw a checkmark at the specified position."""
        # Checkmark shape: short line down-left, then longer line down-right
        pt1 = (cx - size, cy)
        pt2 = (cx - size//3, cy + size//2)
        pt3 = (cx + size, cy - size//2)
        cv2.line(frame, pt1, pt2, (0, 0, 0), thickness + 2)  # Black outline
        cv2.line(frame, pt2, pt3, (0, 0, 0), thickness + 2)
        cv2.line(frame, pt1, pt2, color, thickness)  # Green checkmark
        cv2.line(frame, pt2, pt3, color, thickness)

    def _draw_thumbnails(self, frame):
        """Draw captured photo thumbnails."""
        h, w = frame.shape[:2]
        thumb_size = 60
        spacing = 10
        start_x = spacing
        y = h - thumb_size - spacing - 30
        
        for i, captured in enumerate(self.captured_frames):
            x = start_x + i * (thumb_size + spacing)
            
            # Resize and draw thumbnail
            thumb = cv2.resize(captured, (thumb_size, thumb_size))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)  # Convert back for drawing
            frame[y:y+thumb_size, x:x+thumb_size] = thumb_rgb
            
            # Green border with black outline
            cv2.rectangle(frame, (x-3, y-3), (x+thumb_size+3, y+thumb_size+3), (0, 0, 0), 1)  # Black outline
            cv2.rectangle(frame, (x-2, y-2), (x+thumb_size+2, y+thumb_size+2), (0, 255, 0), 2)
            
            # Draw checkmark (replaces Unicode ✓ which doesn't render in OpenCV)
            self._draw_checkmark(frame, x + thumb_size - 12, y + 12, size=8, color=(0, 255, 0), thickness=2)

    def _do_capture(self, frame):
        """Capture the current frame."""
        logger.info(f"📸 _do_capture called for step {self.current_step + 1}")
        
        # Store the RAW frame (no UI overlays) for DeepFace processing
        with self.frame_lock:
            if self.raw_frame is not None:
                captured = self.raw_frame.copy()  # Use raw frame, not current_frame!
                self.captured_frames.append(captured)
                logger.info(f"✅ RAW frame captured! Shape: {captured.shape}, dtype: {captured.dtype}")
                logger.info(f"   Total captured frames: {len(self.captured_frames)}")
            else:
                logger.warning("⚠️ raw_frame is None - nothing captured!")
        
        self.current_step += 1
        logger.info(f"   Moving to step {self.current_step}")
        
        # Reset advisory and stability state for next capture
        self.last_advisory_audio_time = 0
        self.advisory_audio_count = 0
        self.stable_frame_count = 0
        
        if self.current_step >= len(CAPTURE_TARGETS):
            logger.info("🎉 All captures complete! Starting enrollment...")
            self.capture_complete = True
            self.completion_time = time.time()  # Record when completion happened
            # Trigger enrollment in background
            threading.Thread(target=self._process_enrollment, daemon=True).start()

    def _process_enrollment(self):
        """Process the captured frames for enrollment."""
        logger.info("="*50)
        logger.info("🔄 Starting _process_enrollment")
        logger.info(f"   User name: '{self.user_name}'")
        logger.info(f"   Captured frames count: {len(self.captured_frames)}")
        
        if not self.user_name:
            logger.error("❌ No user name provided!")
            self.enrollment_result = "❌ No name provided"
            return
        
        if len(self.captured_frames) == 0:
            logger.error("❌ No frames were captured!")
            self.enrollment_result = "❌ No frames captured"
            return
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"   Output directory: {OUTPUT_DIR}")
        
        # Save debug images to see what was captured
        debug_dir = os.path.join(OUTPUT_DIR, f"{self.user_name}_debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        embeddings = []
        for i, frame in enumerate(self.captured_frames):
            logger.info(f"   Processing frame {i+1}/{len(self.captured_frames)}")
            logger.info(f"      Frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            try:
                # Save debug image
                debug_path = os.path.join(debug_dir, f"frame_{i+1}.jpg")
                # Raw frame is already BGR from OpenCV
                cv2.imwrite(debug_path, frame)
                logger.info(f"      Saved debug image: {debug_path}")
                
                # Raw frame is already BGR - use directly for DeepFace
                bgr_frame = frame  # Already BGR from cv2.VideoCapture
                logger.info(f"      Calling DeepFace.represent with detector={CHOSEN_DETECTOR}...")
                
                try:
                    # First try with enforce_detection=True
                    result = DeepFace.represent(
                        img_path=bgr_frame,
                        model_name=CHOSEN_MODEL,
                        detector_backend=CHOSEN_DETECTOR,
                        enforce_detection=True,
                        align=True
                    )
                except ValueError:
                    # MediaPipe already confirmed face exists, so retry without strict detection
                    logger.warning(f"      ⚠️ Detector failed, retrying with enforce_detection=False...")
                    result = DeepFace.represent(
                        img_path=bgr_frame,
                        model_name=CHOSEN_MODEL,
                        detector_backend="skip",  # Skip detection, use whole image
                        enforce_detection=False,
                        align=False
                    )
                
                embeddings.append(result[0]["embedding"])
                logger.info(f"      ✅ Face detected! Embedding length: {len(result[0]['embedding'])}")
            except Exception as e:
                logger.error(f"      ❌ DeepFace failed: {type(e).__name__}: {e}")
        
        logger.info(f"   Total successful embeddings: {len(embeddings)}/{len(self.captured_frames)}")
        
        if len(embeddings) >= 2:
            master_embedding = np.mean(embeddings, axis=0)
            
            # Normalize embedding for cosine similarity matching
            norm = np.linalg.norm(master_embedding)
            normalized_embedding = master_embedding / norm if norm > 0 else master_embedding
            
            # Save to SQLite database (safe - no code execution risk like pickle)
            db_path = os.path.join(OUTPUT_DIR, "faces.db")
            init_face_database(db_path)
            
            save_face_embedding(
                db_path=db_path,
                name=self.user_name,
                model=CHOSEN_MODEL,
                detector=CHOSEN_DETECTOR,
                embedding=master_embedding,
                embedding_normalized=normalized_embedding,
                image_count=len(embeddings)
            )
            
            logger.info(f"✅ Enrollment saved to: {db_path}")
            logger.info(f"   User: {self.user_name}")
            logger.info(f"   Embedding shape: {master_embedding.shape}")
            self.enrollment_result = f"✅ SUCCESS! {self.user_name} enrolled with {len(embeddings)} photos."
        else:
            logger.error(f"❌ Not enough valid faces: {len(embeddings)}")
            self.enrollment_result = f"❌ Failed - only {len(embeddings)} valid faces detected."
        
        logger.info("="*50)

    def get_frame_and_status(self):
        """Get current frame and status for Gradio."""
        with self.frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
        
        if self.capture_complete:
            status = self.enrollment_result or "Processing..."
        elif self.current_step >= len(CAPTURE_TARGETS):
            status = "Finalizing..."
        else:
            target = CAPTURE_TARGETS[self.current_step]
            status = f"{target['icon']} {target['name']}: {target['instruction']}"
        
        progress = f"{len(self.captured_frames)}/{len(CAPTURE_TARGETS)} photos"
        
        return frame, status, progress, self.capture_complete, self.enrollment_result


# Global instance - use camera index 0 for USB webcam
# Change this if your webcam is on a different index
WEBCAM_INDEX = 0  # Usually 0 for USB webcam, 2 for CSI on Jetson

# Global RTSP/camera connection parameters (can be overridden via command-line args)
CAMERA_IP = None
PORT = 554
STREAM = "sub"
USER = "admin"
PASSWORD = "Fanatec2025"

capture_system = GuidedEnrollmentCapture(camera_index=WEBCAM_INDEX, camera_ip=CAMERA_IP, port=PORT, stream=STREAM, user=USER, password=PASSWORD)


def start_enrollment(first_name, last_name):
    """Start the enrollment process."""
    first_name = (first_name or "").strip()
    last_name = (last_name or "").strip()
    
    if not first_name:
        return (
            gr.update(visible=True),   # welcome screen stays
            gr.update(visible=False),  # camera screen hidden
            "⚠️ Please enter your first name!"
        )
    
    if not last_name:
        return (
            gr.update(visible=True),   # welcome screen stays
            gr.update(visible=False),  # camera screen hidden
            "⚠️ Please enter your last name!"
        )
    
    # Combine first and last name
    full_name = f"{first_name} {last_name}"
    
    capture_system.reset_capture()
    capture_system.set_user_name(full_name)
    capture_system.start_camera()
    
    return (
        gr.update(visible=False),  # hide welcome
        gr.update(visible=True),   # show camera
        ""
    )


def go_back():
    """Go back to welcome screen."""
    capture_system.stop_camera()
    capture_system.reset_capture()
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        ""
    )


def get_camera_feed():
    """Poll camera feed and thumbnails."""
    frame, status, progress, complete, result = capture_system.get_frame_and_status()
    
    # Get thumbnails (convert BGR to RGB for Gradio display)
    thumbnails = [None, None, None, None, None]
    for i, captured in enumerate(capture_system.captured_frames):
        if i < 5:
            # captured frames are BGR from OpenCV, convert to RGB for Gradio
            thumbnails[i] = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    
    return frame, status, progress, thumbnails[0], thumbnails[1], thumbnails[2], thumbnails[3], thumbnails[4]


AUTO_REVERT_SECONDS = 10  # Auto-revert to welcome screen after this many seconds

def check_completion():
    """Check if enrollment is complete. Returns (done_btn, completion_msg, welcome_screen, camera_screen, error_msg)."""
    if capture_system.capture_complete and capture_system.enrollment_result:
        # Check if auto-revert timer has elapsed
        elapsed = time.time() - capture_system.completion_time
        if elapsed >= AUTO_REVERT_SECONDS:
            # Auto-revert to welcome screen
            logger.info(f"⏱️ Auto-reverting to welcome screen after {AUTO_REVERT_SECONDS}s")
            capture_system.stop_camera()
            capture_system.reset_capture()
            return (
                gr.update(visible=False),  # Hide done button
                gr.update(visible=False, value=""),  # Hide message
                gr.update(visible=True),   # Show welcome screen
                gr.update(visible=False),  # Hide camera screen
                ""  # Clear error message
            )
        
        # Show remaining time in button
        remaining = int(AUTO_REVERT_SECONDS - elapsed)
        return (
            gr.update(visible=True, value=f"✅ Complete Enrollment ({remaining}s)"),  # Show done button with countdown
            gr.update(visible=True, value=capture_system.enrollment_result),  # Show message
            gr.update(),  # No change to welcome screen
            gr.update(),  # No change to camera screen
            gr.update()   # No change to error message
        )
    return (
        gr.update(visible=False), 
        gr.update(visible=False, value=""),
        gr.update(),  # No change to welcome screen
        gr.update(),  # No change to camera screen
        gr.update()   # No change to error message
    )


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Build Blocks with version-compatible arguments
def create_blocks():
    """Create Gradio Blocks with version-compatible settings."""
    import gradio
    version = getattr(gradio, '__version__', '3.0.0')
    major_version = int(version.split('.')[0])
    
    # Gradio 4.x supports theme and css in Blocks
    if major_version >= 4:
        return gr.Blocks(
            title="Face Enrollment",
            theme=gr.themes.Soft(primary_hue="blue", neutral_hue="neutral").set(
                body_background_fill="*neutral_950",
                body_background_fill_dark="*neutral_950",
                block_background_fill="*neutral_900",
                block_background_fill_dark="*neutral_900",
            ),
            css="""
                /* Force dark mode colors consistently across all browsers */
                :root, .dark, .gradio-container {
                    --body-background-fill: #111111 !important;
                    --background-fill-primary: #1a1a1a !important;
                    --background-fill-secondary: #222222 !important;
                    --block-background-fill: #1a1a1a !important;
                    --color-text-primary: #ffffff !important;
                    --color-text-secondary: #cccccc !important;
                    color-scheme: dark !important;
                }
                body, .gradio-container {
                    background-color: #111111 !important;
                    color: #ffffff !important;
                }
                .block, .form, .panel {
                    background-color: #1a1a1a !important;
                }
                /* Ensure markdown text is visible */
                .markdown-text, .prose, .md, p, li, span {
                    color: #e0e0e0 !important;
                }
                h1, h2, h3, h4, h5, h6 {
                    color: #ffffff !important;
                }
                /* Fix label styling across browsers */
                label, .label-wrap, .label-wrap span, .block-label, .block-label span {
                    color: #3b82f6 !important;
                    background-color: #3b82f6 !important;
                    border-radius: 4px !important;
                    padding: 2px 8px !important;
                }
                /* Input field styling */
                input, textarea, .textbox {
                    background-color: #2a2a2a !important;
                    color: #ffffff !important;
                    border-color: #444444 !important;
                }
                .big-button { font-size: 1.5em !important; }
                .big-complete-button { 
                    font-size: 1.8em !important; 
                    padding: 20px 40px !important;
                    min-width: 300px !important;
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
                    border: none !important;
                    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4) !important;
                }
                .big-complete-button:hover {
                    transform: scale(1.05) !important;
                    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.6) !important;
                }
                footer { display: none !important; }
                .gradio-container footer { display: no !important; }
            """,
            js="""
            function() {
                // Force dark theme on page load
                if (window.location.search.indexOf('__theme=dark') === -1) {
                    const url = new URL(window.location);
                    url.searchParams.set('__theme', 'dark');
                    window.location.replace(url.toString());
                }
            }
            """
        )
    else:
        # Older versions - use minimal args
        return gr.Blocks()
# Convert logo to base64 for inline markdown display
logo_path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
logo_base64 = get_image_base64(logo_path)

with create_blocks() as demo:

    # ========================================================================
    # SCREEN 1: Welcome / Name Entry
    # ========================================================================
    with gr.Column(visible=True, elem_classes=["welcome-container"]) as welcome_screen:
        gr.HTML(f"""
        <div style="display:flex; align-items:center; gap:15px; margin-bottom:20px; justify-content: center;">
            <img src="{logo_base64}" alt="QRyde Logo" class="logo" style="height:150px; margin-right:12%;" />
            <h1 style="margin:0; font-size:2em;">Welcome to QRyde Face Enrollmen`t</h1>
        </div>
        """)
        gr.Markdown("""
            <p style="
                color: #ffffff; 
                font-size: 1.4em; 
                font-weight: bold; 
                text-align: center;
                text-shadow: -1px -1px 0 #10421B, 1px -1px 0 #10421B, -1px 1px 0 #10421B, 1px 1px 0 #10421B;
            ">
                To help with product development, we would like to capture your face and use it to train a model to recognize you.
            </p>

            <p style="font-size:1.2em; font-weight:bold; text-align: center;">
                This will take <strong>5 quick photos</strong> cof your face from different angles to set up your profile.
            </p> 
            
            <hr style="border-color: #444;">
            
            <h3 style="color: #000000; margin-bottom: 10px;">How it works:</h3>
            <ol style="color: #000000; size: 1.1em; line-height: 1.8;">
                <li><strong style="color: #5bc0de;">Enter your first and last name</strong> below</li>
                <li><strong style="color: #5bc0de;">Follow the on-screen prompts</strong> - just move your head as instructed</li>
                <li><strong style="color: #5bc0de;">Hold still</strong> when the border turns green</li>
                <li>The camera will <strong style="color: #5bc0de;">automatically capture</strong> after a 3-second countdown</li>
            </ol>
            
            <hr style="border-color: #444;">
            """)
        
        with gr.Row():
            first_name_input = gr.Textbox(
                label="First Name",
                placeholder="Enter your first name...",
                scale=1,
                elem_id="first-name-input"
            )
            last_name_input = gr.Textbox(
                label="Last Name",
                placeholder="Enter your last name...",
                scale=1,
                elem_id="last-name-input"
            )
        
        error_msg = gr.Markdown("", elem_id="error-msg")
        
        start_btn = gr.Button(
            "📸 Start Enrollment",
            variant="primary",
            size="lg",
            elem_classes=["big-button"]
        )
        
        gr.Markdown("""
        ---
        *Press Enter or click the button to begin*
        """)
    
    # ========================================================================
    # SCREEN 2: Camera Capture
    # ========================================================================
    with gr.Column(visible=False) as camera_screen:
        gr.Markdown("## 📷 Face Capture", elem_id="camera-title")
        
        with gr.Row():
            back_btn = gr.Button("← Back", size="sm", variant="secondary")
            done_btn = gr.Button(
                "✅ Complete Enrollment", 
                size="lg", 
                variant="primary", 
                visible=False,
                elem_classes=["big-complete-button"]
            )
            progress_display = gr.Markdown("0/5 photos", elem_classes=["status-box"])
        
        status_display = gr.Markdown(
            "Position your face in the frame...",
            elem_classes=["status-box"]
        )
        
        camera_feed = gr.Image(
            label="",
            height=480,
            elem_classes=["camera-view"],
            show_label=False
        )
        
        # Thumbnail row below video
        gr.Markdown("### Captured Photos")
        with gr.Row():
            thumb_1 = gr.Image(label="1. Front", height=120, show_label=True, visible=True)
            thumb_2 = gr.Image(label="2. Left", height=120, show_label=True, visible=True)
            thumb_3 = gr.Image(label="3. Right", height=120, show_label=True, visible=True)
            thumb_4 = gr.Image(label="4. Up", height=120, show_label=True, visible=True)
            thumb_5 = gr.Image(label="5. Down", height=120, show_label=True, visible=True)
        
        # Completion message (hidden until done)
        completion_msg = gr.Markdown("", visible=False)
        
        # Timer to poll camera
        camera_timer = gr.Timer(value=0.05)  # 20 FPS polling
        completion_timer = gr.Timer(value=0.5)  # Check completion
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    # Start button / Enter key
    start_btn.click(
        fn=start_enrollment,
        inputs=[first_name_input, last_name_input],
        outputs=[welcome_screen, camera_screen, error_msg]
    )
    
    # Allow Enter key on either name field to start
    first_name_input.submit(
        fn=start_enrollment,
        inputs=[first_name_input, last_name_input],
        outputs=[welcome_screen, camera_screen, error_msg]
    )
    
    last_name_input.submit(
        fn=start_enrollment,
        inputs=[first_name_input, last_name_input],
        outputs=[welcome_screen, camera_screen, error_msg]
    )
    
    # Back button
    back_btn.click(
        fn=go_back,
        outputs=[welcome_screen, camera_screen, error_msg]
    )
    
    # Done button
    done_btn.click(
        fn=go_back,
        outputs=[welcome_screen, camera_screen, error_msg]
    )
    
    # Camera feed timer
    camera_timer.tick(
        fn=get_camera_feed,
        outputs=[camera_feed, status_display, progress_display, 
                 thumb_1, thumb_2, thumb_3, thumb_4, thumb_5]
    )
    
    # Completion check timer (also handles auto-revert after 10 seconds)
    completion_timer.tick(
        fn=check_completion,
        outputs=[done_btn, completion_msg, welcome_screen, camera_screen, error_msg]
    )


def open_browser_fullscreen(url, delay=2.0, browser_pref="auto"):
    """Open browser in fullscreen/kiosk mode after a delay.
    
    Args:
        url: URL to open
        delay: Seconds to wait before launching (for server startup)
        browser_pref: Browser preference - "firefox", "chrome", "chromium", or "auto"
    """
    import subprocess
    import shutil
    
    time.sleep(delay)  # Wait for server to start
    
    # Browser configurations
    browser_configs = {
        "firefox": [("firefox", ["--kiosk", url])],
        "chrome": [
            ("google-chrome", ["--kiosk", "--no-first-run", url]),
            ("google-chrome", ["--start-fullscreen", "--no-first-run", url]),
        ],
        "chromium": [
            ("chromium-browser", ["--kiosk", "--no-first-run", url]),
            ("chromium", ["--kiosk", "--no-first-run", url]),
            ("chromium-browser", ["--start-fullscreen", "--no-first-run", url]),
        ],
    }
    
    # Build browser list based on preference
    if browser_pref == "auto":
        browsers = (
            browser_configs["firefox"] + 
            browser_configs["chrome"] + 
            browser_configs["chromium"]
        )
    else:
        browsers = browser_configs.get(browser_pref, [])
    
    for browser, args in browsers:
        if shutil.which(browser):
            try:
                subprocess.Popen([browser] + args)
                logger.info(f"Launched {browser} in fullscreen mode")
                return
            except Exception as e:
                logger.warning(f"Failed to launch {browser}: {e}")
    
    # Fallback to default browser (not fullscreen)
    import webbrowser
    webbrowser.open(url)
    logger.info("Opened default browser (fullscreen not available)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="QRyde Face Enrollment System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kiosk", "-k",
        action="store_true",
        help="Launch browser in kiosk/fullscreen mode"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7861,
        help="Port to run the server on (default: 7861)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index to use (default: 0)"
    )
    parser.add_argument(
        "--camera-ip",
        type=str,
        default=None,
        help="Camera IP address for RTSP stream (default: None, uses local camera)"
    )
    parser.add_argument(
        "--rtsp-port",
        type=int,
        default=554,
        help="RTSP port for camera stream (default: 554)"
    )
    parser.add_argument(
        "--rtsp-stream",
        type=str,
        default="sub",
        help="RTSP stream name (default: sub)"
    )
    parser.add_argument(
        "--rtsp-user",
        type=str,
        default="admin",
        help="RTSP username (default: admin)"
    )
    parser.add_argument(
        "--rtsp-password",
        type=str,
        default="Fanatec2025",
        help="RTSP password (default: Fanatec2025)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link"
    )
    parser.add_argument(
        "--browser",
        type=str,
        choices=["firefox", "chrome", "chromium", "auto"],
        default="auto",
        help="Browser to use in kiosk mode (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Set log level based on argument
    log_level = getattr(logging, args.loglevel.upper())
    logging.getLogger(__name__).setLevel(log_level)
    logger.info(f"Log level set to {args.loglevel}")
    
    # Update global RTSP/camera parameters from args
    CAMERA_IP = args.camera_ip
    PORT = args.rtsp_port
    STREAM = args.rtsp_stream
    USER = args.rtsp_user
    PASSWORD = args.rtsp_password
    
    # Update camera index if specified
    if args.camera != 0:
        WEBCAM_INDEX = args.camera
        logger.info(f"Using camera index: {args.camera}")
    
    # Recreate capture_system with updated parameters
    capture_system = GuidedEnrollmentCapture(
        camera_index=WEBCAM_INDEX, 
        camera_ip=CAMERA_IP, 
        port=PORT, 
        stream=STREAM, 
        user=USER, 
        password=PASSWORD
    )
    logger.info(f"Camera system initialized: camera_ip={CAMERA_IP}, port={PORT}, stream={STREAM}, user={USER}")
    if CAMERA_IP:
        logger.info(f"RTSP URL: {capture_system.rtsp_url}")
    
    logger.info("Starting enrollment system")

    
    url = f"http://localhost:{args.port}"
    
    # Launch browser in kiosk mode if requested
    if args.kiosk:
        threading.Thread(
            target=open_browser_fullscreen, 
            args=(url, 2.0, args.browser), 
            daemon=True
        ).start()
        logger.info(f"Launching in kiosk mode with browser preference: {args.browser}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.kiosk  # Auto-open browser if not in kiosk mode
    )

    logger.info("Enrollment system started")
    
