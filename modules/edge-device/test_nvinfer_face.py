#!/usr/bin/env python3
"""
Test YOLOv8n-face with DeepStream nvinfer element
"""
import sys
for syspath in ['/usr/lib/python3/dist-packages', '/usr/lib/python3.10/dist-packages']:
    if syspath not in sys.path:
        sys.path.append(syspath)

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time

Gst.init(None)

# Simple pipeline: filesrc -> decode -> nvinfer -> fakesink
# For now, let's just test if nvinfer can load the model

RTSP_URL = "rtsp://admin:Fanatec2025@192.168.13.119/Preview_01_sub"
CONFIG_FILE = "config/config_infer_yolov8n_face.txt"

pipeline_str = f'''
    rtspsrc location="{RTSP_URL}" latency=100 ! 
    rtph264depay ! h264parse ! nvv4l2decoder ! 
    m.sink_0 nvstreammux name=m batch-size=1 width=640 height=640 !
    nvinfer config-file-path={CONFIG_FILE} !
    fakesink sync=false
'''

print(f"Testing pipeline with config: {CONFIG_FILE}")
print(f"Pipeline: {pipeline_str}")

try:
    pipeline = Gst.parse_launch(pipeline_str)
    print("Pipeline created successfully")
    
    # Start pipeline
    ret = pipeline.set_state(Gst.State.PLAYING)
    print(f"Set state to PLAYING: {ret}")
    
    # Wait a bit
    time.sleep(10)
    
    # Check state
    success, state, pending = pipeline.get_state(Gst.CLOCK_TIME_NONE)
    print(f"Pipeline state: {state}")
    
    # Stop
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
