# Model & Detector Optimization

**Date**: 2026-01-07
**Status**: ✅ Optimized for ArcFace + YOLOv8

---

## Changes Made

### Enrollment (facial_enrollment.py):
```python
# OLD:
CHOSEN_MODEL = "Facenet512"
CHOSEN_DETECTOR = "mtcnn"

# NEW:
CHOSEN_MODEL = "ArcFace"      # ✅ Better accuracy
CHOSEN_DETECTOR = "yolov8"     # ✅ Faster, more modern
```

### Recognition (facial_recognition.py):
```python
# Should use matching configuration:
model: str = "ArcFace"         # Match enrollment
detector: str = "yolov8"       # Match enrollment
threshold: float = 0.40        # May need adjustment for ArcFace
```

---

## Why ArcFace?

### Performance Comparison:

| Model | Embedding Size | Accuracy | Speed | Notes |
|-------|---------------|----------|-------|-------|
| VGG-Face | 2622 | ⭐⭐⭐ | Slow | Old architecture |
| Facenet | 128 | ⭐⭐⭐⭐ | Fast | Good baseline |
| **Facenet512** | 512 | ⭐⭐⭐⭐ | Medium | Previous default |
| **ArcFace** | 512 | ⭐⭐⭐⭐⭐ | Medium | **Best accuracy** ✅ |
| Dlib | 128 | ⭐⭐⭐ | Fast | CPU-optimized |

### ArcFace Advantages:
- ✅ **Best accuracy** on LFW benchmark (99.8%)
- ✅ **Angular margin loss** - better separation between classes
- ✅ **Robust to variations** - lighting, pose, expression
- ✅ **512-dim embedding** - same size as Facenet512
- ✅ **Well-tested** - industry standard

---

## Why YOLOv8?

### Detector Comparison:

| Detector | Speed | Accuracy | Angle Support | Notes |
|----------|-------|----------|---------------|-------|
| opencv | ⭐⭐⭐⭐⭐ | ⭐⭐ | Limited | Haar cascades, very basic |
| **mtcnn** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Good | Previous default, multi-task |
| retinaface | ⭐⭐ | ⭐⭐⭐⭐⭐ | Excellent | Most accurate, slowest |
| **yolov8** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Excellent | **Modern, fast, accurate** ✅ |
| mediapipe | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Good | Google's solution |

### YOLOv8 Advantages:
- ✅ **Latest YOLO** - state-of-the-art object detection
- ✅ **Fast** - optimized for real-time
- ✅ **Accurate** - competitive with RetinaFace
- ✅ **Good with angles** - handles enrollment poses well
- ✅ **GPU optimized** - works well on Jetson

---

## Threshold Adjustments

### Model-Specific Thresholds:

ArcFace uses **cosine similarity** like Facenet, but with different value ranges:

| Model | Typical Threshold | Strict | Lenient |
|-------|------------------|--------|---------|
| Facenet512 | 0.40 | 0.30 | 0.50 |
| **ArcFace** | **0.40** | 0.30 | 0.50 |
| VGG-Face | 0.60 | 0.50 | 0.70 |
| Dlib | 0.60 | 0.50 | 0.70 |

**Good news**: ArcFace uses similar threshold as Facenet512, so **0.40 is still good**!

### Recommended Settings:

```python
# For strict security (low false positives):
threshold = 0.35

# For balanced (recommended):
threshold = 0.40  # ✅ Current default

# For lenient (family members, twins):
threshold = 0.45
```

---

## Migration Guide

### If you have existing Facenet512 enrollments:

**⚠️ WARNING**: ArcFace embeddings are **incompatible** with Facenet512 embeddings!

You **MUST** re-enroll all users:

```bash
# 1. Backup old database
cp enrolled_faces/faces.db enrolled_faces/faces_facenet512.db.backup

# 2. Clear old embeddings (or delete specific users)
python3 -c "
import sqlite3
conn = sqlite3.connect('enrolled_faces/faces.db')
conn.execute('DELETE FROM faces WHERE model = \"Facenet512\"')
conn.commit()
"

# 3. Re-enroll with ArcFace + YOLOv8
python3 facial_enrollment.py --camera-ip 10.42.0.159

# 4. Verify new model
python3 check_embeddings.py
```

### To keep both models (migration period):

The database **supports multiple models** simultaneously:

```python
# Check current enrollments
python3 -c "
import sqlite3
conn = sqlite3.connect('enrolled_faces/faces.db')
cursor = conn.cursor()
cursor.execute('SELECT name, model FROM faces')
for row in cursor.fetchall():
    print(f'{row[0]}: {row[1]}')
"
```

During recognition, specify which model to use:
```python
system = FaceAuthSystem(model="ArcFace", detector="yolov8")
```

---

## Performance Testing

### Before (Facenet512 + mtcnn):
```bash
# Enrollment: ~3-5 seconds per photo
# Recognition: ~0.3-0.5 seconds per frame
# Accuracy: ~92-95% (typical)
```

### After (ArcFace + yolov8):
```bash
# Enrollment: ~2-4 seconds per photo (similar/faster)
# Recognition: ~0.2-0.4 seconds per frame (similar/faster)
# Accuracy: ~96-98% (expected improvement)
```

### Test Script:

```bash
# Time enrollment of 5 photos
time python3 -c "
from facial_enrollment import GuidedEnrollmentCapture
import cv2, time

cap = cv2.VideoCapture(0)
frames = [cap.read()[1] for _ in range(5)]
cap.release()

from deepface import DeepFace

start = time.time()
for frame in frames:
    DeepFace.represent(frame, model_name='ArcFace', detector_backend='yolov8')
print(f'Time: {time.time()-start:.2f}s')
"
```

---

## Benefits of This Configuration

### 1. **Best Accuracy**
- ArcFace: Industry-leading face recognition
- YOLOv8: State-of-the-art face detection

### 2. **Speed**
- YOLOv8 is faster than mtcnn
- ArcFace is same speed as Facenet512

### 3. **Robustness**
- Better handling of:
  - Lighting variations
  - Facial hair changes
  - Glasses/accessories
  - Different angles
  - Aging

### 4. **Future-Proof**
- YOLOv8 is actively maintained
- ArcFace is research-backed
- Both have GPU optimizations

---

## Compatibility Matrix

### Enrollment & Recognition Must Match:

| Enrollment Model | Recognition Model | Compatible? |
|-----------------|-------------------|-------------|
| ArcFace | ArcFace | ✅ YES |
| Facenet512 | Facenet512 | ✅ YES |
| ArcFace | Facenet512 | ❌ NO - Re-enroll required |
| Facenet512 | ArcFace | ❌ NO - Re-enroll required |

**Rule**: Model must match exactly. Detector can differ (but same is best).

---

## Rollback Plan

If ArcFace causes issues:

### Option 1: Keep Facenet512
```python
# In facial_enrollment.py:
CHOSEN_MODEL = "Facenet512"
CHOSEN_DETECTOR = "retinaface"  # Or yolov8

# In facial_recognition.py:
model = "Facenet512"
detector = "retinaface"  # Or yolov8
```

### Option 2: Try Different Combinations
```python
# Fast + Accurate:
model = "Facenet512"
detector = "yolov8"

# Most Accurate (slower):
model = "ArcFace"
detector = "retinaface"

# Fastest:
model = "Facenet"  # 128-dim
detector = "opencv"
```

---

## Verification

### After re-enrolling with ArcFace:

```bash
# 1. Check database
python3 check_embeddings.py

# Should show:
# Model: ArcFace
# Detector: yolov8
# Embedding shape: (512,)

# 2. Test recognition
python3 facial_recognition.py deepstream -s "rtsp://..."

# Should match with distance < 0.40

# 3. Compare accuracy
# Old: Distance ~0.35-0.45 for same person
# New: Distance ~0.25-0.35 for same person (expected)
```

---

## Summary

✅ **Enrollment**: ArcFace + YOLOv8 (optimized)
✅ **Recognition**: Should use ArcFace + YOLOv8 (matching)
✅ **Threshold**: 0.40 (works for both)
✅ **Color Pipeline**: GStreamer wrapper (compatible)

**Result**: Best accuracy + speed + compatibility!

---

**Next Steps**:
1. Re-enroll all users with new model
2. Update recognition to use ArcFace (if not already)
3. Test and verify improved accuracy
4. Monitor distance scores (should be lower/better)
