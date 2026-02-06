# Enrollment DB & Recognition Pipeline Comparison

Comparison of `enrollments.db`, `enrollments1.db` (old retinaface), and the current recognition pipeline. Use this to avoid reusing incompatible embeddings when switching detectors (e.g. retinaface → yolov8n-face).

---

## 1. Database schemas

| Source | Table | Notes |
|--------|--------|--------|
| **enrollments.db** | (none or different) | In `edge-device/data`, `enrolled_users` may not exist yet (DB not initialized or different schema). |
| **enrollments1.db** | `enrolled_users` | Same schema as `iot_integration/schemas/db_schemas.py`. |
| **Current code** | `enrolled_users` | `user_id`, `display_name`, `model`, `detector`, `embedding` (BLOB), `embedding_dim`, `sync_version`, `synced_at`, `created_at`. |

So the **table layout** is the same for enrollments1 and current code; the important difference is **how** embeddings were produced (detector/alignment).

---

## 2. Stored metadata (enrollments1.db – “old” version)

From `enrollments1.db`:

- **model**: `ArcFace`
- **detector**: `retinaface`
- **embedding_dim**: `512`
- **embedding blob**: `2048` bytes (512 × 4)

So old enrollments are **ArcFace** vectors built from faces that were **detected and aligned with RetinaFace**.

---

## 3. Current recognition pipeline

- **Config** (`config.json`):  
  `recognition.detector_backend": "yolov8n"`, `recognition.model`: `ArcFace`.  
  Detection engine: `yolov8n-face.engine` (TensorRT) when using the TRT path; DeepFace’s `yolov8n` when using DeepFace for detection.

- **Detection** (`main.py`):  
  `FaceRecognitionPipeline.detect_faces()` uses **DeepFace.extract_faces(**…**, detector_backend=self.detector_backend, align=True)**. So detection backend is **yolov8n** (from config), not retinaface.

- **Recognition input**:  
  The pipeline **does not** use the aligned face image returned by `extract_faces`. It only uses the **bounding box** (`facial_area`), then crops from the **original frame** with a margin:
  - `face_crop = frame[y1:y2, x1:x2]` (bbox + 20% margin).
  - That crop is passed to **DeepFace.represent(face_crop, model_name="ArcFace", detector_backend="skip")**.

So at runtime:

1. Faces are **detected** with **yolov8n** (and possibly aligned inside DeepFace for detection).
2. The **crop** given to ArcFace is a **simple bbox crop from the original frame** (no landmark-based alignment in our code).
3. **Embedding** is always **ArcFace** (512-d).

---

## 4. Why detector/alignment matters (and why old embeddings are incompatible)

- **Same model, different “face image”**  
  ArcFace is the same in both cases (512-d). The problem is the **input image** to ArcFace:
  - **Old enrollments (enrollments1.db):**  
    RetinaFace **detects + aligns** the face (e.g. by landmarks), then DeepFace/ArcFace is run on that **aligned** face. So embeddings are from **RetinaFace-aligned** crops.
  - **Current recognition:**  
    Face is detected with **yolov8n**; we only use the **bbox + margin** on the original frame. So ArcFace sees an **unaligned** (or differently aligned) crop.

- **Effect**  
  Same person → different crop (alignment and bbox shape) → **different 512-d vector** → worse matching (higher distance, more false rejects or need for a looser threshold).

- **Conclusion**  
  Embeddings in **enrollments1.db** (retinaface) are **not** compatible with the current **yolov8n-face** recognition path. They should **not** be reused; enrollments should be **recreated** with the same pipeline (same detector and same way of cropping/aligning) as used at recognition.

---

## 5. Potential problem areas in facial matching

1. **Mixing detectors in one DB**  
   `get_all_embeddings()` / `get_all_enrollments()` load **all** rows from `enrolled_users` and do **not** filter by `model` or `detector`. If you add new enrollments with `yolov8n` while keeping old ones with `retinaface`, all are compared against the same query embedding. Retinaface-era embeddings will not match the yolov8n-era crop distribution → inconsistent or poor matching.

2. **No runtime check that DB matches pipeline**  
   The pipeline does not check that stored `model`/`detector` match the current `recognition.model` and `recognition.detector_backend`. Using an old DB (or a copied enrollments1.db) as-is can silently degrade accuracy.

3. **Alignment mismatch (already summarized)**  
   Enrollment: RetinaFace-aligned face → ArcFace.  
   Recognition: yolov8n bbox crop (no landmark alignment) → ArcFace.  
   This is the main reason re-creating embeddings is required when moving to yolov8n-face.

4. **Threshold and distance**  
   `distance_threshold` (e.g. 0.55) is tuned for a consistent enrollment/recognition setup. Mixing detectors can shift the distribution of distances and make the same threshold too strict or too loose.

---

## 6. Recommendations

1. **Do not reuse embeddings from enrollments1.db** for the current yolov8n-face pipeline. Re-create all enrollments using the **same** detector (and same crop/alignment behavior) as at recognition time.
2. **Filter by model + detector when loading**: `DatabaseManager.get_all_enrollments(model_filter=..., detector_filter=...)` and `get_all_embeddings(...)` accept optional `detector_filter`. Pass the current pipeline’s model and detector (e.g. `ArcFace`, `yolov8n` or `yolov8n-face`) so only compatible embeddings are loaded and retinaface-era rows are ignored.
3. **Document detector in enrollment tools**: When writing new rows (enroll script, sync, etc.), always set `detector` to the value that matches the recognition pipeline (e.g. `yolov8n` or `yolov8n-face`) so that the DB stays consistent and filterable later.

---

## 7. Quick reference

| Item | enrollments1.db (old) | Current pipeline |
|------|------------------------|-------------------|
| **Detector** | retinaface | yolov8n / yolov8n-face |
| **Alignment** | RetinaFace landmark alignment | Bbox crop + margin only |
| **Model** | ArcFace | ArcFace |
| **Embedding dim** | 512 | 512 |
| **Reuse old embeddings?** | No | N/A (recreate for yolov8n-face) |
