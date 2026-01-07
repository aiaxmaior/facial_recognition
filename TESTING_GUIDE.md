# Testing Guide - RTSP Color Pipeline Fix

Quick guide to verify the RTSP color pipeline fix is working correctly.

---

## Step 1: Check Color Pipeline Consistency

This test compares OpenCV's default RTSP handling vs the new GStreamer pipeline.

```bash
cd /home/qdrive/facial_recognition
python3 test_rtsp_color.py
```

### ✅ Success Criteria:
- Mean absolute difference < 15.0
- No "green tint" warnings
- Both frames look normal in `/tmp/*.jpg`

### ❌ If it fails:
```bash
# Check GStreamer plugins
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 rtspsrc

# Check OpenCV GStreamer support
python3 -c "import cv2; print('GStreamer' in cv2.getBuildInformation())"
```

---

## Step 2: Check Enrollment Database

This analyzes enrolled users and identifies potential issues.

```bash
python3 check_embeddings.py
```

### ✅ Success Criteria:
- All users have 5 images
- Model is "Facenet512"
- Detector is "mtcnn" or "retinaface"
- Pairwise distances > 0.40 between different people

### ⚠️ If warnings appear:
- **Old enrollments**: Re-enroll with new pipeline
- **Wrong model**: Re-enroll (database has wrong model)
- **Too few images**: Re-enroll with more photos

---

## Step 3: Re-enroll Users (CRITICAL)

**⚠️ IMPORTANT**: Old embeddings used YUV color space, new pipeline uses BGR. You **MUST** re-enroll all users!

### For RTSP camera:
```bash
python3 facial_enrollment.py --camera-ip 10.42.0.159 --rtsp-password Fanatec2025
```

### For USB/local camera:
```bash
python3 facial_enrollment.py --camera 0
```

### During enrollment, check logs:
```bash
# Should see this message:
# "Using GStreamer pipeline for RTSP: rtsp://..."
```

### ✅ Success Criteria:
- Gradio shows normal colors (no green tint)
- 5 photos captured successfully
- "✅ SUCCESS! [Name] enrolled with 5 photos"

---

## Step 4: Test Recognition

### Method A: DeepStream (production path)
```bash
python3 facial_recognition.py deepstream -s "rtsp://admin:Fanatec2025@10.42.0.159/Preview_01_sub"
```

### Method B: Gradio interface
```bash
python3 facial_recognition.py --interface
```
Then navigate to the "Live Stream" tab.

### ✅ Success Criteria:
- Face detected
- Name matches enrolled user
- Distance < 0.40
- "✅ MATCH" displayed

### ❌ If recognition fails:

#### Check logs for:
```
First frame received: shape=(480, 640, 3), dtype=uint8, mean=XXX
```
- **Mean < 50**: Frame too dark, adjust lighting
- **Mean > 200**: Frame too bright, reduce exposure
- **Mean 100-130**: Normal ✓

#### Check distance scores:
```
⛔ Unknown (best guess: [Name], distance: 0.45)
```
- **Distance 0.40-0.50**: Try threshold 0.45 or 0.50
- **Distance > 0.50**: Re-enroll with better lighting/angles

---

## Step 5: Verify Color Consistency (Advanced)

Save a frame during enrollment and recognition, compare pixel values:

```bash
# During enrollment, frames are saved to:
enrolled_faces/[Name]_debug/frame_*.jpg

# During recognition, capture a frame:
# (Add this to facial_recognition.py temporarily)
cv2.imwrite("/tmp/recognition_frame.jpg", frame)

# Compare with ImageMagick:
compare enrolled_faces/[Name]_debug/frame_1.jpg /tmp/recognition_frame.jpg /tmp/diff.png
```

### ✅ Success: Diff image is mostly black (< 5% visible difference)

---

## Troubleshooting Matrix

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Green screen in Gradio | YUV→BGR mismatch | ✅ Fixed by new pipeline |
| "Failed to open stream" | RTSP unreachable | Check camera IP/password |
| Distance > 0.40 for enrolled user | Old embeddings | Re-enroll all users |
| Distance > 0.60 for enrolled user | Bad lighting/angle | Improve conditions, re-enroll |
| Pairwise distance < 0.40 | Similar faces confused | Normal if relatives/twins |
| "GStreamer pipeline failed" | Missing plugins | Install `gstreamer1.0-plugins-bad` |
| Frame mean < 50 | Too dark | Add lighting |
| Frame mean > 200 | Overexposed | Reduce camera gain/exposure |

---

## Quick Reference: Command Summary

```bash
# 1. Test color pipeline
python3 test_rtsp_color.py

# 2. Check database
python3 check_embeddings.py

# 3. Re-enroll users
python3 facial_enrollment.py --camera-ip 10.42.0.159

# 4. Test recognition
python3 facial_recognition.py deepstream -s "rtsp://admin:Fanatec2025@10.42.0.159/Preview_01_sub"

# 5. View logs with debug info
python3 facial_enrollment.py --loglevel DEBUG --camera-ip 10.42.0.159
```

---

## Expected Timeline

- **Step 1** (color test): ~30 seconds
- **Step 2** (check DB): ~5 seconds
- **Step 3** (re-enroll): ~2 minutes per user
- **Step 4** (test recognition): ~1 minute
- **Step 5** (verify): ~2 minutes

**Total**: ~5-10 minutes per user

---

## Success Checklist

Before marking this issue as resolved:

- [ ] test_rtsp_color.py shows diff < 15.0
- [ ] check_embeddings.py shows no warnings
- [ ] All users re-enrolled with new pipeline
- [ ] Gradio shows normal colors (no green)
- [ ] Logs show "Using GStreamer pipeline"
- [ ] Recognition matches users with distance < 0.40
- [ ] Backup created (facial_enrollment.py.backup)

---

## Rollback Plan

If the fix causes new problems:

```bash
# 1. Restore old enrollment script
cp facial_enrollment.py.backup facial_enrollment.py

# 2. Keep new embeddings or restore old ones?
# (Old embeddings won't work with new recognition pipeline!)

# 3. Report issue with:
#    - Logs from enrollment
#    - Output of test_rtsp_color.py
#    - Screenshot of Gradio display
```

---

## Files Created/Modified

- ✅ `facial_enrollment.py` - Modified (backup: `.backup`)
- ✅ `test_rtsp_color.py` - New diagnostic tool
- ✅ `check_embeddings.py` - New diagnostic tool
- ✅ `RTSP_FIX_SUMMARY.md` - Technical documentation
- ✅ `TESTING_GUIDE.md` - This file

---

**Last Updated**: 2026-01-07
**Status**: Ready for testing
