#!/usr/bin/env python3
"""
Facial Recognition with IoT Integration

This script runs the existing facial recognition system with IoT event
transmission to a central broker. It's a drop-in enhancement that adds:
- Event validation across multiple frames
- IoT broker event transmission  
- Compressed image attachments
- Cooldown to prevent duplicate events

Usage:
    # Start with IoT enabled (requires broker URL)
    python run_with_iot.py --device-id cam-001 --broker-url https://iot-broker.example.com
    
    # Start in offline/dev mode (no broker, just validation)
    python run_with_iot.py --device-id cam-001 --offline
    
    # Launch Gradio interface with IoT
    python run_with_iot.py --interface --device-id cam-001 --broker-url https://broker.example.com

    # Adjust validation settings
    python run_with_iot.py --device-id cam-001 --confirmation-frames 3 --cooldown 15
"""

import argparse
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facial_recognition import FaceAuthSystem, GradioInterface, LiveStreamRecognizer
from iot_integration import IoTAdapter, LiveStreamWithIoT, IoTClientConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_live_stream_with_iot(args):
    """Run live stream recognition with IoT event transmission."""
    
    logger.info("="*60)
    logger.info("Facial Recognition with IoT Integration")
    logger.info("="*60)
    
    # Initialize existing face system
    face_system = FaceAuthSystem(
        db_folder=args.db_folder,
        model=args.model,
        detector=args.detector,
        threshold=args.threshold
    )
    
    # Warmup model
    logger.info("Loading facial recognition model...")
    face_system.warmup_model()
    
    # Check enrolled faces
    enrolled = face_system.list_enrolled()
    logger.info(f"Enrolled faces: {len(enrolled)}")
    for entry in enrolled:
        emp_id = entry.get('employee_id', 'N/A')
        logger.info(f"  - {entry['name']} (employee_id: {emp_id})")
    
    if not enrolled:
        logger.warning("No faces enrolled! Use facial_enrollment.py to enroll first.")
    
    # Create IoT-enabled live stream
    broker_url = args.broker_url if not args.offline else "http://localhost:9999"
    
    live_stream = LiveStreamWithIoT(
        face_system=face_system,
        device_id=args.device_id,
        broker_url=broker_url,
        api_key=args.api_key,
        camera_index=args.camera,
        enable_iot=not args.offline,
        recognition_interval=args.recognition_interval,
    )
    
    # Configure validation settings
    live_stream.adapter.event_validator.confirmation_frames = args.confirmation_frames
    live_stream.adapter.event_validator.consistency_threshold = args.consistency_threshold
    live_stream.adapter.event_validator.cooldown_seconds = args.cooldown
    
    # Event callback for logging
    def on_event_sent(event):
        logger.info(f"🚀 IoT Event Sent: user_id={event.user_id}, confidence={event.confidence:.2f}")
    
    live_stream.adapter.on_event_sent = on_event_sent
    
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Device ID: {args.device_id}")
    logger.info(f"  Broker URL: {broker_url}")
    logger.info(f"  IoT Enabled: {not args.offline}")
    logger.info(f"  Camera: {args.camera}")
    logger.info(f"  Recognition Interval: {args.recognition_interval}s")
    logger.info(f"  Confirmation Frames: {args.confirmation_frames}")
    logger.info(f"  Consistency Threshold: {args.consistency_threshold}")
    logger.info(f"  Cooldown: {args.cooldown}s")
    logger.info("")
    
    # Start stream
    result = live_stream.start()
    logger.info(result)
    
    if "Failed" in result:
        return
    
    # Run OpenCV display loop
    import cv2
    import numpy as np
    
    logger.info("Press 'q' to quit, 's' for stats")
    logger.info("")
    
    try:
        while True:
            frame, status = live_stream.get_frame()
            
            if frame is not None:
                # Convert RGB back to BGR for OpenCV display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add status bar at bottom
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0, h-30), (w, h), (40, 40, 40), -1)
                cv2.putText(display_frame, status[:80], (10, h-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.imshow('Facial Recognition + IoT', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Print stats
                stats = live_stream.get_stats()
                print("\n" + "="*40)
                print("IoT Statistics:")
                print(f"  Events validated: {stats['validator'].get('events_validated', 0)}")
                print(f"  Events rejected (consistency): {stats['validator'].get('events_rejected_consistency', 0)}")
                print(f"  Events rejected (cooldown): {stats['validator'].get('events_rejected_cooldown', 0)}")
                print(f"  Active tracks: {stats.get('active_tracks', 0)}")
                if 'client' in stats:
                    print(f"  Events sent: {stats['client'].get('events_sent', 0)}")
                    print(f"  Events failed: {stats['client'].get('events_failed', 0)}")
                print("="*40 + "\n")
    
    except KeyboardInterrupt:
        logger.info("\nStopping...")
    finally:
        live_stream.stop()
        cv2.destroyAllWindows()
    
    # Final stats
    stats = live_stream.get_stats()
    logger.info("")
    logger.info("Final Statistics:")
    logger.info(f"  Total recognitions processed: {stats['validator'].get('recognitions_processed', 0)}")
    logger.info(f"  Events validated & sent: {stats['validator'].get('events_validated', 0)}")


def run_gradio_with_iot(args):
    """Run Gradio interface with IoT-enabled live stream."""
    import gradio as gr
    
    logger.info("Launching Gradio interface with IoT integration...")
    
    # Initialize face system
    face_system = FaceAuthSystem(
        db_folder=args.db_folder,
        model=args.model,
        detector=args.detector,
        threshold=args.threshold
    )
    
    # Create IoT-enabled live stream
    broker_url = args.broker_url if not args.offline else "http://localhost:9999"
    
    live_stream = LiveStreamWithIoT(
        face_system=face_system,
        device_id=args.device_id,
        broker_url=broker_url,
        api_key=args.api_key,
        camera_index=args.camera,
        enable_iot=not args.offline,
    )
    
    # Configure validation
    live_stream.adapter.event_validator.confirmation_frames = args.confirmation_frames
    live_stream.adapter.event_validator.cooldown_seconds = args.cooldown
    
    # Build Gradio interface
    with gr.Blocks(title="Facial Recognition + IoT") as demo:
        gr.Markdown(f"""
        # 🎯 Facial Recognition with IoT Integration
        
        **Device ID:** `{args.device_id}` | **IoT:** `{'Enabled' if not args.offline else 'Offline'}` | 
        **Model:** `{args.model}` | **Threshold:** `{args.threshold}`
        
        Events are validated across **{args.confirmation_frames} frames** before transmission.
        Cooldown: **{args.cooldown}s** between same-user events.
        """)
        
        with gr.Tab("🎬 Live Stream + IoT"):
            with gr.Row():
                start_btn = gr.Button("▶️ Start", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                stats_btn = gr.Button("📊 Stats", size="lg")
            
            status_box = gr.Textbox(label="Status", value="Click Start to begin")
            video_feed = gr.Image(label="Live Feed", height=480)
            result_box = gr.Textbox(label="Recognition Result")
            stats_box = gr.JSON(label="IoT Statistics", visible=False)
            
            def start():
                return live_stream.start()
            
            def stop():
                return live_stream.stop()
            
            def get_frame():
                return live_stream.get_frame()
            
            def get_stats():
                stats = live_stream.get_stats()
                return gr.update(value=stats, visible=True)
            
            start_btn.click(fn=start, outputs=[status_box])
            stop_btn.click(fn=stop, outputs=[status_box])
            stats_btn.click(fn=get_stats, outputs=[stats_box])
            
            timer = gr.Timer(value=0.1)
            timer.tick(fn=get_frame, outputs=[video_feed, result_box])
        
        with gr.Tab("👥 Enrolled Employees"):
            enrolled_list = gr.Dataframe(
                headers=["Name", "Employee ID", "Email", "Enrolled"],
                label="Enrolled Faces"
            )
            refresh_btn = gr.Button("🔄 Refresh")
            
            def get_enrolled():
                enrolled = face_system.list_enrolled()
                rows = []
                for e in enrolled:
                    rows.append([
                        f"{e.get('first_name', '')} {e.get('last_name', '')}".strip() or e.get('name'),
                        e.get('employee_id', 'N/A'),
                        e.get('email', 'N/A'),
                        e.get('enrolled_at', 'Unknown')[:10] if e.get('enrolled_at') else 'Unknown'
                    ])
                return rows
            
            refresh_btn.click(fn=get_enrolled, outputs=[enrolled_list])
            demo.load(fn=get_enrolled, outputs=[enrolled_list])
        
        with gr.Tab("⚙️ IoT Settings"):
            gr.Markdown(f"""
            ### Current Configuration
            
            | Setting | Value |
            |---------|-------|
            | Device ID | `{args.device_id}` |
            | Broker URL | `{broker_url}` |
            | Confirmation Frames | `{args.confirmation_frames}` |
            | Consistency Threshold | `{args.consistency_threshold}` |
            | Cooldown (seconds) | `{args.cooldown}` |
            | Recognition Interval | `{args.recognition_interval}s` |
            
            To change settings, restart with different command-line arguments.
            """)
    
    # Launch
    face_system.warmup_model()
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Facial Recognition with IoT Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument(
        "--device-id", "-d",
        type=str,
        default="cam-001",
        help="Unique device identifier (default: cam-001)"
    )
    
    # IoT settings
    parser.add_argument(
        "--broker-url", "-b",
        type=str,
        default="http://localhost:8080",
        help="IoT broker URL"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for IoT broker"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run without IoT transmission (validation only)"
    )
    
    # Validation settings
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=5,
        help="Frames required to confirm identity (default: 5)"
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.8,
        help="Required consistency ratio (default: 0.8)"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Seconds between same-user events (default: 30)"
    )
    
    # Face recognition settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="ArcFace",
        help="Face recognition model (default: ArcFace)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolov8",
        help="Face detector backend (default: yolov8)"
    )
    parser.add_argument(
        "--threshold", "-T",
        type=float,
        default=0.40,
        help="Matching threshold (default: 0.40)"
    )
    parser.add_argument(
        "--db-folder",
        type=str,
        default="enrolled_faces",
        help="Folder for enrolled faces"
    )
    
    # Camera/interface settings
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--recognition-interval",
        type=float,
        default=0.5,
        help="Seconds between recognition attempts (default: 0.5)"
    )
    parser.add_argument(
        "--interface", "-I",
        action="store_true",
        help="Launch Gradio web interface"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Gradio server port (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    
    args = parser.parse_args()
    
    if args.interface:
        run_gradio_with_iot(args)
    else:
        run_live_stream_with_iot(args)


if __name__ == "__main__":
    main()
