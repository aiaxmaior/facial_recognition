#!/usr/bin/env python3
"""
GStreamer-based camera capture wrapper for RTSP streams.
Provides OpenCV-compatible interface using subprocess and GStreamer CLI.
This allows enrollment to use the SAME pipeline as DeepStream recognition.
"""
import subprocess
import numpy as np
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class GStreamerCamera:
    """
    Camera capture using GStreamer subprocess.
    Provides cv2.VideoCapture-like interface for RTSP streams.
    """

    def __init__(self, rtsp_url: str, width: int = 640, height: int = 480):
        """
        Initialize GStreamer camera capture.

        Args:
            rtsp_url: RTSP stream URL
            width: Target frame width
            height: Target frame height
        """
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.process = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None

        # Build GStreamer pipeline (matches DeepStream exactly)
        self.pipeline = (
            f'rtspsrc location={rtsp_url} latency=100 ! '
            'rtph264depay ! h264parse ! '
            'nvv4l2decoder ! '
            'nvvideoconvert ! video/x-raw,format=BGRx ! '
            'videoconvert ! video/x-raw,format=BGR ! '
            f'videoscale ! video/x-raw,width={width},height={height} ! '
            'fdsink fd=1'  # Output to stdout
        )

    def isOpened(self) -> bool:
        """Check if camera is successfully opened."""
        return self.running and self.process is not None and self.process.poll() is None

    def open(self) -> bool:
        """Start the GStreamer pipeline."""
        if self.running:
            return True

        try:
            logger.info(f"Starting GStreamer pipeline: {self.rtsp_url}")

            # Start GStreamer process
            self.process = subprocess.Popen(
                ['gst-launch-1.0', '-q'] + self.pipeline.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.width * self.height * 3
            )

            self.running = True

            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            # Wait for first frame to verify pipeline works
            time.sleep(1.0)

            if not self.isOpened():
                logger.error("GStreamer pipeline failed to start")
                return False

            logger.info("✅ GStreamer pipeline started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start GStreamer: {e}")
            return False

    def read(self):
        """
        Read a frame from the camera.

        Returns:
            (ret, frame) tuple where ret is bool and frame is numpy array
        """
        if not self.isOpened():
            return False, None

        try:
            # Get latest frame from queue (non-blocking with timeout)
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        """Stop the GStreamer pipeline and cleanup."""
        self.running = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        logger.info("GStreamer camera released")

    def set(self, prop_id, value):
        """
        Dummy set method for compatibility with cv2.VideoCapture.
        GStreamer properties are set in pipeline, so this is a no-op.
        """
        # Width/height are set in pipeline construction
        # FPS, buffer size, etc. are fixed in pipeline
        pass

    def _capture_loop(self):
        """Background thread to capture frames from GStreamer stdout."""
        frame_size = self.width * self.height * 3  # BGR format

        logger.debug(f"Capture thread started, expecting {frame_size} bytes per frame")

        while self.running and self.process and self.process.poll() is None:
            try:
                # Read exactly one frame worth of data
                raw_data = self.process.stdout.read(frame_size)

                if len(raw_data) != frame_size:
                    logger.warning(f"Incomplete frame: got {len(raw_data)}, expected {frame_size}")
                    continue

                # Convert to numpy array
                frame = np.frombuffer(raw_data, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))

                # Put in queue, drop old frames if queue is full
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(frame, block=False)

            except Exception as e:
                if self.running:
                    logger.error(f"Frame capture error: {e}")
                break

        logger.debug("Capture thread ended")


# Convenience function to create camera
def create_rtsp_camera(rtsp_url: str, width: int = 640, height: int = 480):
    """
    Create and open a GStreamer-based RTSP camera.

    Args:
        rtsp_url: RTSP stream URL
        width: Frame width
        height: Frame height

    Returns:
        GStreamerCamera instance (already opened)
    """
    camera = GStreamerCamera(rtsp_url, width, height)
    if camera.open():
        return camera
    else:
        raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")


if __name__ == "__main__":
    # Test the camera
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 gstreamer_camera.py <rtsp_url>")
        print("Example: python3 gstreamer_camera.py rtsp://admin:pass@192.168.1.100/stream")
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG)

    rtsp_url = sys.argv[1]
    print(f"Testing GStreamer camera with: {rtsp_url}")

    cam = GStreamerCamera(rtsp_url, 640, 480)

    if not cam.open():
        print("❌ Failed to open camera")
        sys.exit(1)

    print("✅ Camera opened, capturing frames...")
    print("Press Ctrl+C to stop")

    frame_count = 0
    try:
        while True:
            ret, frame = cam.read()
            if ret:
                frame_count += 1
                print(f"Frame {frame_count}: shape={frame.shape}, mean={frame.mean():.2f}")

                # Save first frame as test
                if frame_count == 1:
                    import cv2
                    cv2.imwrite("/tmp/gstreamer_test.jpg", frame)
                    print("  Saved: /tmp/gstreamer_test.jpg")

                time.sleep(0.1)
            else:
                print("Failed to read frame")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.release()
        print(f"Captured {frame_count} frames total")
