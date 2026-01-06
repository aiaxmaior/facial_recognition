# Facial Recognition System
Author: Arjun Joshi
Date: 12.8.2025

A comprehensive, multi-purpose facial recognition system powered by [DeepFace](https://github.com/serengil/deepface). 
Supports CLI usage, Python module import, and an optional Gradio web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.90+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Face Enrollment** - Register faces with averaged embeddings from multiple images
- **Guided Enrollment System** - Elder-friendly interface with MediaPipe head pose detection and audio guidance
- **Face Matching** - Identify faces against the enrollment database
- **Live Camera Verification** - Capture 5 photos over 10 seconds with averaged matching
- **RTSP Camera Support** - Connect to IP cameras via RTSP streams (Reolink, etc.)
- **DeepFace Stream** - Real-time continuous face recognition
- **DeepStream Integration** - Hardware-accelerated video processing for Jetson devices
- **Direct Verification** - Compare two faces using `DeepFace.verify()`
- **Facial Attribute Analysis** - Detect age, gender, race, and emotion
- **Admin Portal** - Web-based database management tool for enrolled faces
- **SQLite Database** - Safe, efficient storage of face embeddings (no code execution risk)
- **Audio Guidance** - Spoken instructions and feedback during enrollment
- **Kiosk Mode** - Fullscreen browser interface for dedicated enrollment stations
- **Multiple Interfaces** - CLI, Python module, or Gradio web UI

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/aiaxmaior/facial_recognition.git
cd facial_recognition

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- OpenCV
- DeepFace
- TensorFlow
- Gradio (optional, for web interface)
- MediaPipe (for guided enrollment)
- Pygame (for audio guidance)
- SQLite3 (built-in with Python)

### Jetson/NVIDIA Setup

For NVIDIA Jetson devices (Orin Nano, Xavier, etc.) with DeepStream support:

1. **Install DeepStream SDK** (see [reolink_jetson_setup.md](reolink_jetson_setup.md) for detailed instructions):
   ```bash
   sudo apt update
   sudo apt install -y deepstream-6.3 python3-gi python3-gst-1.0 \
       gir1.2-gst-rtsp-server-1.0 libgstreamer1.0-dev \
       libgstreamer-plugins-base1.0-dev
   ```

2. **Set up PyDS Python bindings**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/opt/nvidia/deepstream/deepstream/lib/
   ```

3. **Verify installation**:
   ```bash
   python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print('GStreamer OK')"
   python3 -c "import pyds; print('PyDS OK')"
   ```

See [reolink_jetson_setup.md](reolink_jetson_setup.md) for complete Jetson setup instructions, including RTSP camera configuration.

## 📖 Usage

### Command Line Interface

```bash
# Launch Gradio web interface
python facial_recognition.py --interface

# Enroll a new person (provide multiple images for better accuracy)
python facial_recognition.py enroll -n "John Doe" -i photo1.jpg photo2.jpg photo3.jpg

# Match/identify a face against the database
python facial_recognition.py match -t unknown_person.jpg

# Verify if two images are the same person
python facial_recognition.py verify -1 face1.jpg -2 face2.jpg

# Analyze facial attributes (age, gender, race, emotion)
python facial_recognition.py analyze -i face.jpg

# List all enrolled faces
python facial_recognition.py list

# Delete an enrolled face
python facial_recognition.py delete -n "John Doe"

# DeepStream - RTSP stream processing
python facial_recognition.py deepstream -s "rtsp://192.168.1.100:554/stream"

# DeepStream - USB camera
python facial_recognition.py deepstream -s /dev/video0

# DeepStream - CSI camera (Jetson)
python facial_recognition.py deepstream -s csi

# DeepStream - Video file
python facial_recognition.py deepstream -s video.mp4

# DeepStream - Multiple sources
python facial_recognition.py deepstream-multi -s "rtsp://cam1/stream" "rtsp://cam2/stream"
```

### Guided Enrollment System

The guided enrollment system (`facial_enrollment.py`) provides an elder-friendly interface with automatic head pose detection:

```bash
# Launch guided enrollment interface
python facial_enrollment.py

# With RTSP camera
python facial_enrollment.py --camera-ip 192.168.1.100 --rtsp-port 554 --rtsp-stream sub --rtsp-user admin --rtsp-password YOUR_PASSWORD

# Kiosk mode (fullscreen)
python facial_enrollment.py --kiosk

# Custom port and camera
python facial_enrollment.py --port 7861 --camera 0

# Debug logging
python facial_enrollment.py --loglevel DEBUG
```

**Features:**
- MediaPipe head pose detection guides users to correct positions
- Audio instructions in multiple languages
- Automatic capture when pose is held correctly
- 5-picture enrollment (Front, Left, Right, Up, Down)
- Real-time visual feedback with colored borders
- SQLite database storage

### Admin Portal

Manage the enrolled faces database with the admin tool:

```bash
# Launch interactive CLI
python face_admin.py

# Launch web interface
python face_admin.py --web

# List all enrolled users
python face_admin.py list

# Search for users
python face_admin.py search "John"

# Delete a user
python face_admin.py delete "John Doe"

# Show user details
python face_admin.py info "John Doe"

# Export to CSV
python face_admin.py export users.csv

# Run custom SQL query
python face_admin.py query "SELECT * FROM faces WHERE model='Facenet512'"
```

### RTSP Camera Configuration

Connect to IP cameras (Reolink, Hikvision, etc.) via RTSP:

```bash
# Using facial_enrollment.py with RTSP
python facial_enrollment.py \
    --camera-ip 192.168.1.100 \
    --rtsp-port 554 \
    --rtsp-stream sub \
    --rtsp-user admin \
    --rtsp-password YOUR_PASSWORD

# RTSP URL format: rtsp://user:password@camera_ip:port/stream
# Example: rtsp://admin:password123@192.168.1.100:554/sub
```

The system automatically constructs the RTSP URL from the provided components. See [reolink_jetson_setup.md](reolink_jetson_setup.md) for detailed camera setup instructions.

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--interface`, `-I` | Launch Gradio web interface | - |
| `--model`, `-m` | Recognition model | `Facenet512` |
| `--detector`, `-d` | Face detector backend | `retinaface` |
| `--db-folder` | Enrollment database folder | `enrolled_faces` |
| `--threshold`, `-T` | Matching threshold (lower = stricter) | `0.40` |
| `--port`, `-p` | Gradio server port | `7860` |

### Supported Models

`VGG-Face`, `Facenet`, `Facenet512`, `OpenFace`, `DeepFace`, `DeepID`, `ArcFace`, `Dlib`, `SFace`, `GhostFaceNet`

### Supported Detectors

`opencv`, `ssd`, `dlib`, `mtcnn`, `fastmtcnn`, `retinaface`, `mediapipe`, `yolov8`, `yunet`, `centerface`

## 🌐 Gradio Web Interface

Launch with:
```bash
python facial_recognition.py --interface
```

The web interface includes 6 tabs:

| Tab | Description |
|-----|-------------|
| 📹 **Camera Verification** | Captures 5 photos over 10 seconds, uses averaged embeddings |
| 🎬 **DeepFace Stream** | Real-time continuous recognition via `DeepFace.stream()` |
| 📤 **Manual Verification** | Upload up to 5 images for averaged matching |
| 📝 **Enrollment** | Register new faces + manage enrolled database |
| 🔗 **Direct Verify** | Compare two images directly (demo feature) |
| 🔬 **Facial Analysis** | Analyze age, gender, race, emotion |

## 🐍 Python Module Usage

```python
from facial_recognition import FaceAuthSystem, GradioInterface

# Initialize system
system = FaceAuthSystem(
    db_folder="enrolled_faces",
    model="Facenet512",
    detector="retinaface",
    threshold=0.40
)

# Enroll a person
system.enroll("John Doe", ["img1.jpg", "img2.jpg", "img3.jpg"])

# Match against database
name, distance, is_match = system.match("unknown.jpg")
if is_match:
    print(f"Identified: {name} (distance: {distance:.4f})")

# Direct verification
result = system.verify_images("face1.jpg", "face2.jpg")
print(f"Same person: {result['verified']}")

# Analyze facial attributes
analysis = system.analyze_face("face.jpg")
print(f"Age: {analysis['age']}")
print(f"Gender: {analysis['dominant_gender']}")
print(f"Emotion: {analysis['dominant_emotion']}")
print(f"Race: {analysis['dominant_race']}")

# Launch Gradio interface programmatically
import gradio as gr
interface = GradioInterface(system)
interface.launch(
    server_port=7860,
    theme=gr.themes.Soft()
)
```

## 📁 Project Structure

```
facial_recognition/
├── facial_recognition.py   # Main comprehensive system
├── facesystem_core.py      # Core FaceAuthSystem (standalone)
├── facial_enrollment.py    # Guided enrollment with MediaPipe pose detection
├── face_admin.py           # Admin portal for database management
├── requirements.txt        # Python dependencies
├── reolink_jetson_setup.md # Jetson setup guide
├── reolink_setup.html      # Interactive setup guide
├── .gitignore
├── README.md
├── enrolled_faces/         # SQLite database (faces.db) + debug images
├── audio/                  # Audio guidance files for enrollment
│   ├── *.mp3              # Primary guidance audio
│   └── secondary_guidance/ # Correction/feedback audio
├── images/                 # Logo and UI assets
├── test_images/           # Sample test images
├── comfyui_workflows/     # ComfyUI workflow files
└── Tools/                 # Additional tools
```

## 🔧 How It Works

### Enrollment Process
1. Multiple face images are processed through the detection pipeline
2. Each face is aligned and encoded into a 512-dimensional embedding vector
3. Embeddings are averaged to create a "master" centroid vector
4. The master vector is stored in a **SQLite database** (`faces.db`) along with metadata:
   - Name, model, detector used
   - Normalized embedding for cosine similarity
   - Image count, enrollment timestamp
   - Debug images saved for verification

### Matching Process
1. Target image(s) are processed to extract embedding(s)
2. For multi-image matching, embeddings are averaged
3. Cosine distance is calculated against all enrolled faces using normalized embeddings
4. If the minimum distance is below the threshold, it's a match

### Guided Enrollment (facial_enrollment.py)
1. MediaPipe FaceMesh detects head pose (yaw, pitch)
2. Real-time guidance directs user to correct position
3. Audio instructions provide spoken feedback
4. When pose is held correctly for 0.5 seconds, countdown begins
5. After 3-second countdown, photo is automatically captured
6. Process repeats for 5 different poses (Front, Left, Right, Up, Down)
7. All embeddings are averaged and stored in SQLite database

## 💻 Hardware Requirements

### Standard x86/x64 Systems
- CPU: Multi-core processor (Intel/AMD)
- RAM: 4GB minimum, 8GB recommended
- GPU: Optional (CUDA-compatible for faster processing)
- Camera: USB webcam or IP camera with RTSP support

### NVIDIA Jetson Devices
- **Jetson Orin Nano** (recommended) or **Jetson Xavier NX**
- DeepStream SDK 6.3+
- CUDA-capable GPU for hardware acceleration
- PoE switch for IP camera connectivity (optional)
- See [reolink_jetson_setup.md](reolink_jetson_setup.md) for complete setup

### Camera Options
- **USB Webcam**: Direct connection via `/dev/video0` or camera index
- **RTSP IP Cameras**: Reolink, Hikvision, Axis, etc.
- **CSI Cameras**: Native Jetson camera modules
- **Video Files**: MP4, AVI, etc. for batch processing

## 📚 References

- [DeepFace GitHub](https://github.com/serengil/deepface) - The underlying face recognition library
- [DeepFace Documentation](https://github.com/serengil/deepface#readme)
- [Gradio](https://gradio.app/) - Web interface framework
- [MediaPipe](https://mediapipe.dev/) - Face mesh and pose detection
- [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) - Video analytics SDK
- [Reolink Jetson Setup Guide](reolink_jetson_setup.md) - Camera configuration guide

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.


---

*Built with [DeepFace](https://github.com/serengil/deepface) • Powered by deep learning*

