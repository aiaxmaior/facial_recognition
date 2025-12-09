# Facial Recognition System
Author: Arjun Joshi
Date: 12.8.2025

Summary:
A comprehensive, multi-purpose facial recognition system powered by [DeepFace]
Supports CLI usage, Python module import, and an optional Gradio web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.90+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Face Enrollment** - Register faces with averaged embeddings from multiple images
- **Face Matching** - Identify faces against the enrollment database
- **Live Camera Verification** - Capture 5 photos over 10 seconds with averaged matching
- **DeepFace Stream** - Real-time continuous face recognition
- **Direct Verification** - Compare two faces using `DeepFace.verify()`
- **Facial Attribute Analysis** - Detect age, gender, race, and emotion
- **Multiple Interfaces** - CLI, Python module, or Gradio web UI

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/facial_recognition.git
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
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--interface`, `-I` | Launch Gradio web interface | - |
| `--model`, `-m` | Recognition model | `Facenet512` |
| `--detector`, `-d` | Face detector backend | `retinaface` |
| `--db-folder` | Enrollment database folder | `enrolled_faces` |
| `--threshold`, `-T` | Matching threshold (lower = stricter) | `0.30` |
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
    threshold=0.30
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
├── facial_enrollment.py    # Enrollment-only Gradio app
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── enrolled_faces/         # Enrollment database (gitignored)
├── face_repo/              # Face image repository (gitignored)
└── comfyui_workflows/      # ComfyUI workflow files
```

## 🔧 How It Works

### Enrollment Process
1. Multiple face images are processed through the detection pipeline
2. Each face is aligned and encoded into a 512-dimensional embedding vector
3. Embeddings are averaged to create a "master" centroid vector
4. The master vector is stored as a `.pkl` file in the enrollment database

### Matching Process
1. Target image(s) are processed to extract embedding(s)
2. For multi-image matching, embeddings are averaged
3. Cosine distance is calculated against all enrolled faces
4. If the minimum distance is below the threshold, it's a match

## 📚 References

- [DeepFace GitHub](https://github.com/serengil/deepface) - The underlying face recognition library
- [DeepFace Documentation](https://github.com/serengil/deepface#readme)
- [Gradio](https://gradio.app/) - Web interface framework

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.


---

*Built with [DeepFace](https://github.com/serengil/deepface) • Powered by deep learning*

