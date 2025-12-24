"""
Facial Enrollment System - Simple Elder-Friendly Interface
Uses MediaPipe FaceMesh for automatic head pose detection and guided capture.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import gradio as gr
import threading
import os
import pickle
from deepface import DeepFace

# Configuration
CHOSEN_MODEL = "Facenet512"
OUTPUT_DIR = "enrolled_faces"
COUNTDOWN_SECONDS = 3
POSE_HOLD_TIME = 0.5  # How long to hold pose before countdown starts

# Capture targets - what head orientation we want for each photo
# NORMALIZED: pitch 0 = straight, positive = up, negative = down
# yaw 0 = centered, positive = left, negative = right
# WIDE TOLERANCES for easier capture
CAPTURE_TARGETS = [
    {"name": "Front", "yaw_range": (-25, 25), "pitch_range": (-20, 20), 
     "icon": "Front", "instruction": "Look straight at the camera"},
    {"name": "Left", "yaw_range": (10, 50), "pitch_range": (-30, 30), 
     "icon": "Left", "instruction": "Turn your head LEFT"},
    {"name": "Right", "yaw_range": (-50, -10), "pitch_range": (-30, 30), 
     "icon": "Right", "instruction": "Turn your head RIGHT"},
    {"name": "Up", "yaw_range": (-30, 30), "pitch_range": (-50, -15), 
     "icon": "Up", "instruction": "Tilt your chin UP slightly"},
    {"name": "Down", "yaw_range": (-40, 40), "pitch_range": (15, 50), 
     "icon": "Down", "instruction": "Tilt your chin DOWN slightly"},
]


class GuidedEnrollmentCapture:
    """Handles camera capture with MediaPipe face mesh and auto-detection."""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.camera_ready = False
        self.current_frame = None      # Processed frame (with overlays) for display
        self.raw_frame = None          # Raw frame (no overlays) for capture
        self.frame_lock = threading.Lock()
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
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
        
        # "Locked in" state - once countdown starts, we're more forgiving
        self.locked_yaw = 0
        self.locked_pitch = 0
        self.countdown_tolerance = 40  # Extra degrees of tolerance during countdown
        self.enrollment_result = ""
        
        # Current telemetry
        self.current_yaw = 0
        self.current_pitch = 0
        self.face_detected = False

    def start_camera(self):
        """Start camera capture thread."""
        if self.running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        
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
        self.enrollment_result = ""

    def set_user_name(self, name):
        """Set the user name for enrollment."""
        self.user_name = name.strip().replace(" ", "_")

    def _put_text_outlined(self, frame, text, pos, font, scale, color, thickness):
        """Draw text with outline for visibility."""
        cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness + 3)
        cv2.putText(frame, text, pos, font, scale, color, thickness)

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
            
            # Only process every Nth frame to reduce CPU load
            if frame_count % process_every_n == 0:
                # Store raw frame (flipped, but no overlays) for face capture
                raw = cv2.flip(frame, 1)
                
                processed = self._process_frame(frame)
                
                with self.frame_lock:
                    self.raw_frame = raw.copy()  # Clean frame for DeepFace
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
            
            # Draw face mesh overlay
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
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
        """Draw the capture UI with instructions and countdown."""
        h, w = frame.shape[:2]
        
        if self.capture_complete:
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
                # User moved too much - cancel countdown
                self.countdown_active = False
                self._put_text_outlined(frame, "Hold still! Try again...", (w//2 - 150, h - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
                cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 100, 255), 4)
            else:
                # Still locked in - show countdown
                countdown_num = int(remaining) + 1
                
                # Big countdown number
                self._put_text_outlined(frame, str(countdown_num), (w//2 - 40, h//2 + 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
                
                # Progress ring
                progress = (COUNTDOWN_SECONDS - remaining) / COUNTDOWN_SECONDS
                end_angle = int(360 * progress)
                cv2.ellipse(frame, (w//2, h//2), (100, 100), -90, 0, end_angle, (0, 255, 0), 8)
                
                # Green border during countdown
                cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 4)
        
        elif pose_valid:
            # Pose is correct - start building up to countdown
            if not self.last_pose_valid:
                self.pose_held_since = time.time()
            
            hold_duration = time.time() - self.pose_held_since
            
            if hold_duration >= POSE_HOLD_TIME:
                # Start countdown and LOCK IN current pose
                self.countdown_active = True
                self.countdown_start = time.time()
                self.locked_yaw = yaw
                self.locked_pitch = pitch
            else:
                # Pose correct, waiting to start countdown
                self._put_text_outlined(frame, "HOLD STILL...", (w//2 - 100, h - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Green border
                cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 4)
        else:
            # Pose not correct - show guidance
            if not self.face_detected:
                self._put_text_outlined(frame, "No face detected", (w//2 - 120, h - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 0, 255), 4)
            else:
                # Show guidance arrows
                self._draw_guidance(frame, yaw, pitch, target)
                cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 165, 255), 3)
        
        self.last_pose_valid = pose_valid
        
        # Show captured thumbnails at bottom
        self._draw_thumbnails(frame)
        
        return frame

    def _draw_guidance(self, frame, yaw, pitch, target):
        """Draw arrows to guide user to correct pose."""
        h, w = frame.shape[:2]
        
        # Show current pose values for debugging (small text in corner)
        debug_text = f"Yaw:{yaw:.0f} Pitch:{pitch:.0f}"
        self._put_text_outlined(frame, debug_text, (10, h - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Determine which way to move (use ASCII-safe arrows)
        hints = []
        
        # Yaw guidance (left/right) - positive yaw = looking left
        if yaw < target["yaw_range"][0]:
            hints.append("<-- Turn LEFT")
        elif yaw > target["yaw_range"][1]:
            hints.append("Turn RIGHT -->")
        
        # Pitch guidance (up/down)
        # Normalized: positive pitch = up, negative pitch = down
        if pitch < target["pitch_range"][0]:
            hints.append("Look UP ^")    # Need higher pitch (more positive)
        elif pitch > target["pitch_range"][1]:
            hints.append("Look DOWN v")  # Need lower pitch (more negative)
        
        if hints:
            hint_text = " | ".join(hints)
            # Center the text better
            text_size = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = (w - text_size[0]) // 2
            self._put_text_outlined(frame, hint_text, (text_x, h - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    def _draw_thumbnails(self, frame):
        """Draw captured photo thumbnails."""
        h, w = frame.shape[:2]
        thumb_size = 60
        spacing = 10
        start_x = spacing
        y = h - thumb_size - spacing - 30
        
        for i, captured in enumerate(self.captured_frames):
            x = start_x + i * (thumb_size + spacing)
            
            # Resize and draw thumbnail (captured frames are BGR, same as display frame)
            thumb = cv2.resize(captured, (thumb_size, thumb_size))
            frame[y:y+thumb_size, x:x+thumb_size] = thumb
            
            # Green border
            cv2.rectangle(frame, (x-2, y-2), (x+thumb_size+2, y+thumb_size+2), (0, 255, 0), 2)
            
            # Checkmark
            cv2.putText(frame, "✓", (x + thumb_size - 20, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _do_capture(self, frame):
        """Capture the current frame."""
        # Store the RAW frame (no overlays) for DeepFace processing
        with self.frame_lock:
            if self.raw_frame is not None:
                self.captured_frames.append(self.raw_frame.copy())
        
        self.current_step += 1
        
        if self.current_step >= len(CAPTURE_TARGETS):
            self.capture_complete = True
            # Trigger enrollment in background
            threading.Thread(target=self._process_enrollment, daemon=True).start()

    def _process_enrollment(self):
        """Process the captured frames for enrollment."""
        if not self.user_name:
            self.enrollment_result = "❌ No name provided"
            return
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        embeddings = []
        for i, frame in enumerate(self.captured_frames):
            try:
                # Raw frames are already in BGR format from OpenCV
                result = DeepFace.represent(
                    img_path=frame,
                    model_name=CHOSEN_MODEL,
                    enforce_detection=True,
                    align=True
                )
                embeddings.append(result[0]["embedding"])
            except Exception as e:
                pass  # Skip failed frames
        
        if len(embeddings) >= 2:
            master_embedding = np.mean(embeddings, axis=0)
            
            data = {
                "name": self.user_name,
                "model": CHOSEN_MODEL,
                "embedding": master_embedding,
                "image_count": len(embeddings)
            }
            
            output_path = os.path.join(OUTPUT_DIR, f"{self.user_name}_deepface.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
            
            self.enrollment_result = f"✅ SUCCESS! {self.user_name} enrolled with {len(embeddings)} photos."
        else:
            self.enrollment_result = f"❌ Failed - only {len(embeddings)} valid faces detected."

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
capture_system = GuidedEnrollmentCapture(camera_index=WEBCAM_INDEX)


def start_enrollment(name):
    """Start the enrollment process."""
    if not name or not name.strip():
        return (
            gr.update(visible=True),   # welcome screen stays
            gr.update(visible=False),  # camera screen hidden
            "Please enter your name first!"
        )
    
    capture_system.reset_capture()
    capture_system.set_user_name(name)
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
    """Poll camera feed."""
    frame, status, progress, complete, result = capture_system.get_frame_and_status()
    return frame, status, progress


def check_completion():
    """Check if enrollment is complete."""
    if capture_system.capture_complete and capture_system.enrollment_result:
        return (
            gr.update(visible=True),
            capture_system.enrollment_result
        )
    return gr.update(visible=False), ""


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
            theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
            css=".big-button { font-size: 1.5em !important; }"
        )
    else:
        # Older versions - use minimal args
        return gr.Blocks()

with create_blocks() as demo:
    
    # ========================================================================
    # SCREEN 1: Welcome / Name Entry
    # ========================================================================
    with gr.Column(visible=True, elem_classes=["welcome-container"]) as welcome_screen:
        gr.Markdown("""
        # 👋 Welcome to Face Enrollment
        
        This will take **5 quick photos** of your face from different angles 
        to set up your profile.
        
        ---
        
        ### How it works:
        1. **Enter your name** below
        2. **Follow the on-screen prompts** - just move your head as instructed
        3. **Hold still** when the border turns green
        4. The camera will **automatically capture** after a 3-second countdown
        
        ---
        """)
        
        name_input = gr.Textbox(
            label="Your Name",
            placeholder="Enter your full name...",
            scale=1,
            elem_id="name-input"
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
        
        # Completion panel (hidden until done)
        with gr.Column(visible=False) as completion_panel:
            completion_msg = gr.Markdown("")
            done_btn = gr.Button("✅ Done - Return to Start", variant="primary", size="lg")
        
        # Timer to poll camera
        camera_timer = gr.Timer(value=0.05)  # 20 FPS polling
        completion_timer = gr.Timer(value=0.5)  # Check completion
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    # Start button / Enter key
    start_btn.click(
        fn=start_enrollment,
        inputs=[name_input],
        outputs=[welcome_screen, camera_screen, error_msg]
    )
    
    name_input.submit(
        fn=start_enrollment,
        inputs=[name_input],
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
        outputs=[camera_feed, status_display, progress_display]
    )
    
    # Completion check timer
    completion_timer.tick(
        fn=check_completion,
        outputs=[completion_panel, completion_msg]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
