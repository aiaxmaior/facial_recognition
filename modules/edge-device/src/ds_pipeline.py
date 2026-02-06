"""
DeepStream/GStreamer RTSP Video Pipeline

Hardware-accelerated video decode using nvv4l2decoder on Jetson.
Provides frames to a callback function for processing.

Architecture:
    rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvideoconvert → appsink
                                              ↑
                                    Hardware H.264 decode (NVDEC)
"""

import sys
# Add system site-packages for gi (GObject) when running in venv
# Append (not insert) so venv packages take priority
for syspath in ['/usr/lib/python3/dist-packages', '/usr/lib/python3.10/dist-packages']:
    if syspath not in sys.path:
        sys.path.append(syspath)

import threading
import time
import logging
import numpy as np
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class PipelineStats:
    """Pipeline statistics."""
    frames_received: int = 0
    frames_dropped: int = 0
    last_frame_time: float = 0
    fps: float = 0
    reconnect_count: int = 0
    start_time: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "fps": round(self.fps, 1),
            "reconnect_count": self.reconnect_count,
            "uptime_seconds": round(time.time() - self.start_time, 1) if self.start_time else 0
        }


class DeepStreamPipeline:
    """
    GStreamer pipeline for RTSP video with hardware-accelerated decode.
    
    Usage:
        def on_frame(frame: np.ndarray, timestamp: float):
            # Process frame (640x360 BGR)
            pass
        
        pipeline = DeepStreamPipeline(config)
        pipeline.start(on_frame)
        # ... later ...
        pipeline.stop()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Dict with keys:
                - camera.rtsp_url: RTSP stream URL
                - pipeline.deepstream.latency_ms: Buffer latency (default 100)
                - pipeline.deepstream.output_width: Output width (default 640)
                - pipeline.deepstream.output_height: Output height (default 360)
        """
        Gst.init(None)
        
        self.config = config
        self.rtsp_url = config["camera"]["rtsp_url"]
        
        # DeepStream settings with defaults
        ds_config = config.get("pipeline", {}).get("deepstream", {})
        self.latency_ms = ds_config.get("latency_ms", 100)
        self.output_width = ds_config.get("output_width", 640)
        self.output_height = ds_config.get("output_height", 360)
        self.output_format = ds_config.get("output_format", "BGRx")
        
        # Pipeline state
        self.pipeline: Optional[Gst.Pipeline] = None
        self.main_loop: Optional[GLib.MainLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        
        self.state = PipelineState.STOPPED
        self.stats = PipelineStats()
        self._frame_callback: Optional[Callable] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # FPS calculation
        self._fps_frames = 0
        self._fps_start_time = 0
        
        logger.info(f"DeepStreamPipeline initialized: {self.output_width}x{self.output_height}")
    
    def _build_pipeline(self) -> Gst.Pipeline:
        """Build the GStreamer pipeline."""
        pipeline_str = f'''
            rtspsrc name=source location="{self.rtsp_url}" latency={self.latency_ms} 
                    drop-on-latency=true protocols=tcp buffer-mode=auto !
            rtph264depay !
            h264parse !
            nvv4l2decoder name=decoder enable-max-performance=true !
            nvvideoconvert name=converter !
            video/x-raw,format={self.output_format},width={self.output_width},height={self.output_height} !
            appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false
        '''
        
        logger.debug(f"Pipeline: {pipeline_str}")
        
        pipeline = Gst.parse_launch(pipeline_str)
        
        # Connect appsink signal
        appsink = pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_new_sample)
        
        # Connect bus messages
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        return pipeline
    
    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        """Handle new frame from appsink."""
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        try:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)
            width = struct.get_value("width")
            height = struct.get_value("height")
            
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            
            try:
                # BGRx format: 4 bytes per pixel
                frame = np.ndarray(
                    shape=(height, width, 4),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                # Convert BGRx to BGR (drop alpha channel)
                frame_bgr = frame[:, :, :3].copy()
                
                # Get timestamp
                pts = buf.pts
                timestamp = pts / Gst.SECOND if pts != Gst.CLOCK_TIME_NONE else time.time()
                
                # Update stats
                with self._lock:
                    self.stats.frames_received += 1
                    self.stats.last_frame_time = time.time()
                    self._fps_frames += 1
                    
                    # Calculate FPS every second
                    now = time.time()
                    if now - self._fps_start_time >= 1.0:
                        self.stats.fps = self._fps_frames / (now - self._fps_start_time)
                        self._fps_frames = 0
                        self._fps_start_time = now
                
                # Deliver frame to callback
                if self._frame_callback:
                    try:
                        self._frame_callback(frame_bgr, timestamp)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                        
            finally:
                buf.unmap(map_info)
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
        
        return Gst.FlowReturn.OK
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer error: {err.message}")
            logger.debug(f"Debug info: {debug}")
            self.state = PipelineState.ERROR
            
            # Attempt reconnection
            if not self._stop_event.is_set():
                self._schedule_reconnect()
                
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"GStreamer warning: {warn.message}")
            
        elif msg_type == Gst.MessageType.EOS:
            logger.info("End of stream")
            if not self._stop_event.is_set():
                self._schedule_reconnect()
                
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                if new == Gst.State.PLAYING:
                    logger.info("Pipeline is now PLAYING")
                    self.state = PipelineState.RUNNING
    
    def _schedule_reconnect(self, delay: float = 2.0):
        """Schedule pipeline reconnection after delay."""
        if self._stop_event.is_set():
            return
            
        self.state = PipelineState.RECONNECTING
        self.stats.reconnect_count += 1
        logger.info(f"Scheduling reconnect in {delay}s (attempt #{self.stats.reconnect_count})")
        
        def reconnect():
            if self._stop_event.is_set():
                return False
            self._restart_pipeline()
            return False  # Don't repeat
        
        GLib.timeout_add(int(delay * 1000), reconnect)
    
    def _restart_pipeline(self):
        """Restart the pipeline."""
        logger.info("Restarting pipeline...")
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        
        try:
            self.pipeline = self._build_pipeline()
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to restart pipeline")
                self._schedule_reconnect(delay=5.0)
            else:
                self.state = PipelineState.STARTING
        except Exception as e:
            logger.error(f"Pipeline restart failed: {e}")
            self._schedule_reconnect(delay=5.0)
    
    def _run_main_loop(self):
        """Run GLib main loop in thread."""
        try:
            self.main_loop = GLib.MainLoop()
            self.main_loop.run()
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            logger.debug("Main loop exited")
    
    def start(self, frame_callback: Callable[[np.ndarray, float], None]):
        """
        Start the pipeline.
        
        Args:
            frame_callback: Function called for each frame.
                           Signature: callback(frame: np.ndarray, timestamp: float)
                           Frame is BGR, shape (height, width, 3)
        """
        if self.state != PipelineState.STOPPED:
            logger.warning(f"Pipeline already in state: {self.state}")
            return
        
        logger.info("Starting DeepStream pipeline...")
        self._frame_callback = frame_callback
        self._stop_event.clear()
        
        # Reset stats
        self.stats = PipelineStats()
        self.stats.start_time = time.time()
        self._fps_start_time = time.time()
        
        # Build and start pipeline
        try:
            self.pipeline = self._build_pipeline()
            self.state = PipelineState.STARTING
            
            # Start main loop in thread
            self.loop_thread = threading.Thread(target=self._run_main_loop, daemon=True)
            self.loop_thread.start()
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start pipeline")
            
            logger.info("Pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.state = PipelineState.ERROR
            raise
    
    def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping DeepStream pipeline...")
        self._stop_event.set()
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        
        if self.main_loop:
            self.main_loop.quit()
            self.main_loop = None
        
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=2.0)
        
        self.state = PipelineState.STOPPED
        self._frame_callback = None
        
        logger.info(f"Pipeline stopped. Stats: {self.stats.to_dict()}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        with self._lock:
            return self.stats.to_dict()
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self.state == PipelineState.RUNNING
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# Simple test
if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open("config/config.json") as f:
        config = json.load(f)
    
    frame_count = [0]
    
    def on_frame(frame: np.ndarray, timestamp: float):
        frame_count[0] += 1
        if frame_count[0] % 30 == 0:
            print(f"Frame {frame_count[0]}: shape={frame.shape}, ts={timestamp:.3f}")
    
    pipeline = DeepStreamPipeline(config)
    
    try:
        pipeline.start(on_frame)
        print("Pipeline running. Press Ctrl+C to stop...")
        
        while True:
            time.sleep(5)
            stats = pipeline.get_stats()
            print(f"Stats: {stats}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.stop()
