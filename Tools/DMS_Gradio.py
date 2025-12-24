import cv2
import mediapipe as mp
import numpy as np
import time
import gradio as gr
from collections import deque
import threading

class DMS_Gradio:
    def __init__(self, camera_index=0):
        # OpenCV capture (like DMS.py - much faster than Gradio webcam)
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Drawing Utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Camera Matrix (will be initialized on first frame)
        self.cam_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # 3D Model Points
        self.face_3d = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left Eye Left Corner
            (225.0, 170.0, -135.0),     # Right Eye Right Corner
            (-150.0, -150.0, -125.0),   # Left Mouth Corner
            (150.0, -150.0, -125.0)     # Right Mouth Corner
        ], dtype=np.float64)

        self.face_2d_indices = [1, 152, 33, 263, 61, 291]

        # State tracking
        self.drowsy_frames = 0
        self.distracted_frames = 0
        self.prev_time = time.time()
        self.fps = 0
        self.attention_history = deque(maxlen=30)
        
        # Calibration state
        self.calibration = {
            "calibrated": False,
            "neutral_yaw": 0.0,
            "neutral_pitch": 0.0,
            "neutral_roll": 0.0,
            "baseline_ear": 0.3,
            "samples": [],
            "calibrating": False,
            "calibration_start": 0,
            "calibration_duration": 5.0
        }
        
        # Configurable thresholds
        self.config = {
            "ear_threshold": 0.25,
            "yaw_threshold": 20.0,
            "pitch_down_threshold": -10.0,
            "drowsy_frame_threshold": 10,
            "distraction_frame_threshold": 15,
        }
        
        # Scoring
        self.attention_score = 1.0
        self.drowsiness_score = 0.0
        self.distraction_score = 0.0

    def _put_text_outlined(self, frame, text, pos, font, scale, color, thickness, outline_color=(0, 0, 0), outline_thickness=2):
        """Draw text with black outline for better readability"""
        # Draw outline (black border)
        cv2.putText(frame, text, pos, font, scale, outline_color, thickness + outline_thickness)
        # Draw main text on top
        cv2.putText(frame, text, pos, font, scale, color, thickness)
        
        # Latest telemetry
        self.telemetry = {
            "status": "Initializing...",
            "pitch": 0, "yaw": 0, "roll": 0,
            "ear": 0, "fps": 0,
            "inference": "Starting camera...",
            "attention": "N/A", "drowsiness": "N/A", "distraction": "N/A",
            "calibration": "🔴 Not Calibrated"
        }

    def start_camera(self):
        """Start OpenCV camera capture in background thread"""
        if self.running:
            return "Camera already running"
        
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            return "❌ Failed to open camera"
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return "✅ Camera started"

    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        return "Camera stopped"

    def _capture_loop(self):
        """Background thread for camera capture and processing"""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame
            processed = self._process_frame(frame)
            
            with self.frame_lock:
                self.current_frame = processed

    def get_ear(self, landmarks, refer_idxs, frame_w, frame_h):
        try:
            coords_points = []
            for i in refer_idxs:
                lm = landmarks[i]
                coord = np.array([lm.x * frame_w, lm.y * frame_h])
                coords_points.append(coord)
            
            P2_P6 = np.linalg.norm(coords_points[1] - coords_points[5])
            P3_P5 = np.linalg.norm(coords_points[2] - coords_points[4])
            P1_P4 = np.linalg.norm(coords_points[0] - coords_points[3])
            
            ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
            return ear
        except:
            return 0.0

    def _process_frame(self, frame):
        """Process a single frame - similar to DMS.py"""
        h, w = frame.shape[:2]
        
        # Initialize camera matrix
        if self.cam_matrix is None:
            focal_length = w
            self.cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        status_text = "No Face Detected"
        pitch, yaw, roll = 0.0, 0.0, 0.0
        avg_ear = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the full face mesh tesselation (geometric triangular overlay)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw face contours (eyes, lips, eyebrows, face oval)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # Draw iris outlines (requires refine_landmarks=True)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                # Head Pose
                face_2d = []
                for idx in self.face_2d_indices:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_2d.append([x, y])
                face_2d = np.array(face_2d, dtype=np.float64)

                success_pnp, rot_vec, trans_vec = cv2.solvePnP(
                    self.face_3d, face_2d, self.cam_matrix, self.dist_coeffs
                )

                if success_pnp:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles = cv2.RQDecomp3x3(rmat)[0]
                    
                    pitch = angles[0] * 360
                    yaw = angles[1] * 360
                    roll = angles[2] * 360

                    if yaw > -self.config["yaw_threshold"]:
                        status_text = "Looking Left"
                        self.distracted_frames += 1
                    elif yaw < self.config["yaw_threshold"]:
                        status_text = "Looking Right"
                        self.distracted_frames += 1
                    elif pitch < self.config["pitch_down_threshold"]:
                        status_text = "Looking Down"
                        self.distracted_frames += 1
                    else:
                        status_text = "Forward"
                        self.distracted_frames = max(0, self.distracted_frames - 1)

                    # Nose projection line
                    nose_3d_projection, _ = cv2.projectPoints(
                        self.face_3d[0], rot_vec, trans_vec, self.cam_matrix, self.dist_coeffs
                    )
                    p1 = (int(face_2d[0][0]), int(face_2d[0][1]))
                    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                    cv2.line(frame, p1, p2, (255, 0, 0), 3)

                # EAR calculation
                LEFT_EYE = [362, 385, 387, 263, 373, 380]
                RIGHT_EYE = [33, 160, 158, 133, 153, 144]
                left_ear = self.get_ear(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear = self.get_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < self.config["ear_threshold"]:
                    self.drowsy_frames += 1
                    status_text = "Eyes Closed"
                else:
                    self.drowsy_frames = 0

                if self.drowsy_frames > self.config["drowsy_frame_threshold"]:
                    status_text = "DROWSINESS ALERT!"
                    self._put_text_outlined(frame, "DROWSINESS ALERT!", (w//2 - 150, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # Iris visualization
                left_iris = face_landmarks.landmark[468]
                right_iris = face_landmarks.landmark[473]
                cv2.circle(frame, (int(left_iris.x * w), int(left_iris.y * h)), 4, (0, 255, 255), -1)
                cv2.circle(frame, (int(right_iris.x * w), int(right_iris.y * h)), 4, (0, 255, 255), -1)

        # Calculate FPS
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 30
        self.prev_time = curr_time
        
        # Draw FPS on frame (with outline for readability)
        self._put_text_outlined(frame, f"FPS: {int(self.fps)}", (w - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self._put_text_outlined(frame, f"Status: {status_text}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update scores
        self._update_scores(yaw, pitch, avg_ear)
        
        # Handle calibration
        cal_status = "🔴 Not Calibrated" if not self.calibration["calibrated"] else "🟢 Calibrated"
        if self.calibration["calibrating"]:
            cal_status = self._process_calibration(yaw, pitch, roll, avg_ear)
        
        # Update telemetry
        self.telemetry = {
            "status": status_text,
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "ear": avg_ear, "fps": int(self.fps),
            "inference": self._generate_inference(status_text, yaw, pitch, avg_ear),
            "attention": f"{self.attention_score*100:.0f}%",
            "drowsiness": f"{self.drowsiness_score*100:.0f}%",
            "distraction": f"{self.distraction_score*100:.0f}%",
            "calibration": cal_status
        }
        
        # Convert BGR to RGB for Gradio display
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _update_scores(self, yaw, pitch, avg_ear):
        """Update attention scores"""
        # Distraction
        yaw_dev = abs(yaw) / self.config["yaw_threshold"]
        pitch_dev = max(0, -pitch / abs(self.config["pitch_down_threshold"]))
        self.distraction_score = min(1.0, (yaw_dev * 0.6 + pitch_dev * 0.4))
        
        # Drowsiness
        if self.drowsy_frames > self.config["drowsy_frame_threshold"]:
            self.drowsiness_score = min(1.0, self.drowsiness_score + 0.2)
        else:
            self.drowsiness_score = max(0.0, self.drowsiness_score - 0.05)
        
        # Attention
        self.attention_score = 1.0 - max(self.distraction_score, self.drowsiness_score)
        self.attention_history.append(self.attention_score)
        if len(self.attention_history) > 5:
            self.attention_score = np.mean(list(self.attention_history))

    def _generate_inference(self, status, yaw, pitch, ear):
        """Generate inference comments"""
        comments = []
        
        if status == "No Face Detected":
            return "⚠️ Driver face not visible!"
        
        if self.attention_score > 0.8:
            comments.append("✅ Driver is attentive")
        elif self.attention_score > 0.5:
            comments.append("⚠️ Moderate attention")
        else:
            comments.append("🚨 LOW ATTENTION!")
        
        if self.drowsiness_score > 0.5:
            comments.append("😴 Signs of drowsiness")
        
        if status in ["Looking Left", "Looking Right", "Looking Down"]:
            comments.append(f"👀 {status}")
        
        return " | ".join(comments)

    def _process_calibration(self, yaw, pitch, roll, ear):
        """Process calibration samples"""
        elapsed = time.time() - self.calibration["calibration_start"]
        
        if elapsed < self.calibration["calibration_duration"]:
            self.calibration["samples"].append({"yaw": yaw, "pitch": pitch, "roll": roll, "ear": ear})
            remaining = self.calibration["calibration_duration"] - elapsed
            return f"🔄 Calibrating... {remaining:.1f}s ({len(self.calibration['samples'])} samples)"
        else:
            if len(self.calibration["samples"]) > 10:
                samples = self.calibration["samples"]
                self.calibration["neutral_yaw"] = np.mean([s["yaw"] for s in samples])
                self.calibration["neutral_pitch"] = np.mean([s["pitch"] for s in samples])
                self.calibration["baseline_ear"] = np.mean([s["ear"] for s in samples])
                self.calibration["calibrated"] = True
                self.calibration["calibrating"] = False
                return f"✅ Calibrated! Yaw={self.calibration['neutral_yaw']:.1f}°"
            else:
                self.calibration["calibrating"] = False
                return "❌ Calibration failed - not enough samples"

    def start_calibration(self):
        """Start calibration"""
        self.calibration["calibrating"] = True
        self.calibration["calibration_start"] = time.time()
        self.calibration["samples"] = []
        return "🔄 Look straight ahead for 5 seconds..."

    def update_config(self, ear_thresh, yaw_thresh, pitch_thresh, drowsy_frames):
        """Update config"""
        self.config["ear_threshold"] = ear_thresh
        self.config["yaw_threshold"] = yaw_thresh
        self.config["pitch_down_threshold"] = pitch_thresh
        self.config["drowsy_frame_threshold"] = int(drowsy_frames)
        return f"✓ Config updated"

    def get_frame(self):
        """Get current processed frame for Gradio"""
        with self.frame_lock:
            if self.current_frame is not None:
                return (
                    self.current_frame,
                    self.telemetry["status"],
                    self.telemetry["pitch"],
                    self.telemetry["yaw"],
                    self.telemetry["roll"],
                    self.telemetry["ear"],
                    self.telemetry["fps"],
                    self.telemetry["inference"],
                    self.telemetry["attention"],
                    self.telemetry["drowsiness"],
                    self.telemetry["distraction"],
                    self.telemetry["calibration"]
                )
        return None, "Waiting...", 0, 0, 0, 0, 0, "Starting...", "N/A", "N/A", "N/A", "🔴 Not Started"


# Gradio Interface
dms = DMS_Gradio(camera_index=0)

with gr.Blocks(title="Q-DRIVE DMS Monitor") as demo:
    gr.Markdown("#Q-DRIVE Driver Monitoring System")
    gr.Markdown("*Using OpenCV capture for 30+ FPS*")
    
    with gr.Row():
        start_btn = gr.Button("▶️ Start Camera", variant="primary")
        stop_btn = gr.Button("⏹️ Stop Camera", variant="stop")
        camera_status = gr.Textbox(label="Camera Status", value="Not started")
    
    with gr.Row():
        with gr.Column(scale=2):
            output_video = gr.Image(label="DMS Feed", height=480)
            
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Telemetry")
            status_output = gr.Textbox(label="Driver Status", value="Waiting...")
            calibration_status = gr.Textbox(label="Calibration", value="🔴 Not Calibrated")
            
            with gr.Group():
                gr.Markdown("### 📈 Scores")
                with gr.Row():
                    attention_out = gr.Textbox(label="Attention", value="N/A", scale=1)
                    drowsiness_out = gr.Textbox(label="Drowsiness", value="N/A", scale=1)
                    distraction_out = gr.Textbox(label="Distraction", value="N/A", scale=1)
            
            with gr.Group():
                gr.Markdown("### 🎯 Head Pose")
                with gr.Row():
                    pitch_out = gr.Number(label="Pitch", value=0)
                    yaw_out = gr.Number(label="Yaw", value=0)
                    roll_out = gr.Number(label="Roll", value=0)
            
            with gr.Row():
                ear_out = gr.Number(label="EAR", value=0)
                fps_out = gr.Number(label="FPS", value=0)
    
    gr.Markdown("## 💬 Analysis")
    inference_box = gr.Textbox(label="System Inference", value="Click Start Camera...", lines=2)
    
    with gr.Accordion("⚙️ Configuration", open=False):
        with gr.Row():
            ear_slider = gr.Slider(0.15, 0.35, value=0.25, step=0.01, label="EAR Threshold")
            yaw_slider = gr.Slider(10, 40, value=20, step=1, label="Yaw Threshold (°)")
        with gr.Row():
            pitch_slider = gr.Slider(-30, 0, value=-10, step=1, label="Pitch Threshold (°)")
            drowsy_slider = gr.Slider(5, 30, value=10, step=1, label="Drowsy Frames")
        with gr.Row():
            config_btn = gr.Button("Apply Config")
            calibrate_btn = gr.Button("🎯 Calibrate")
        config_status = gr.Textbox(label="Config Status", value="Default config")
    
    # Event handlers
    start_btn.click(fn=dms.start_camera, outputs=[camera_status])
    stop_btn.click(fn=dms.stop_camera, outputs=[camera_status])
    config_btn.click(
        fn=dms.update_config,
        inputs=[ear_slider, yaw_slider, pitch_slider, drowsy_slider],
        outputs=[config_status]
    )
    calibrate_btn.click(fn=dms.start_calibration, outputs=[calibration_status])
    
    # Polling timer to get frames (much faster than Gradio webcam streaming)
    timer = gr.Timer(value=0.033)  # ~30 FPS polling
    timer.tick(
        fn=dms.get_frame,
        outputs=[output_video, status_output, pitch_out, yaw_out, roll_out, 
                 ear_out, fps_out, inference_box, attention_out, drowsiness_out,
                 distraction_out, calibration_status]
    )

if __name__ == "__main__":
    demo.launch(share=False)
