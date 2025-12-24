from deepface import DeepFace
import numpy as np
import pickle
import gradio as gr
import os
import time
import cv2
import tempfile
from pathlib import Path

# Configuration
CHOSEN_MODEL = "Facenet512"
OUTPUT_DIR = "enrolled_faces"

# Capture prompts for different angles
CAPTURE_PROMPTS = [
    {"angle": "Front", "icon": "👤", "instruction": "Look directly at the camera"},
    {"angle": "Left", "icon": "👈", "instruction": "Turn your head slightly LEFT"},
    {"angle": "Right", "icon": "👉", "instruction": "Turn your head slightly RIGHT"},
    {"angle": "Up", "icon": "👆", "instruction": "Tilt your chin slightly UP"},
    {"angle": "Down", "icon": "👇", "instruction": "Tilt your chin slightly DOWN"},
]


def enroll_person_deepface(name, images):
    """Enroll a person using uploaded images."""
    if not name or not name.strip():
        return "❌ Error: Please enter a name.", None

    if not images or len(images) == 0:
        return "❌ Error: Please upload at least one image.", None

    name = name.strip().replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_messages = [f"--- Enrolling {name} with {CHOSEN_MODEL} ---"]
    known_embeddings = []

    for img in images:
        img_path = img if isinstance(img, str) else img.name
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=CHOSEN_MODEL,
                enforce_detection=True,
                align=True
            )

            embedding = result[0]["embedding"]
            known_embeddings.append(embedding)
            log_messages.append(f"✔ Processed: {os.path.basename(img_path)}")

        except ValueError:
            log_messages.append(f"⚠ Warning: Face could not be detected in {os.path.basename(img_path)}. Skipping.")
        except Exception as e:
            log_messages.append(f"⚠ Error processing {os.path.basename(img_path)}: {e}")

    if len(known_embeddings) > 0:
        master_embedding = np.mean(known_embeddings, axis=0)

        data = {
            "name": name,
            "model": CHOSEN_MODEL,
            "embedding": master_embedding
        }

        output_path = os.path.join(OUTPUT_DIR, f"{name}_deepface.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        log_messages.append(f"\n✅ SUCCESS: {name} enrolled!")
        log_messages.append(f"📊 Vector Size: {len(master_embedding)}")
        log_messages.append(f"📁 Saved to: {output_path}")
        log_messages.append(f"🖼 Images processed: {len(known_embeddings)}/{len(images)}")

        return "\n".join(log_messages), output_path
    else:
        log_messages.append("\n❌ FAILURE: No valid faces found in any image.")
        return "\n".join(log_messages), None


def enroll_from_captured_images(name, img1, img2, img3, img4, img5):
    """Enroll using the 5 captured images from guided capture."""
    if not name or not name.strip():
        return "❌ Error: Please enter a name first."
    
    images = [img1, img2, img3, img4, img5]
    valid_images = [img for img in images if img is not None]
    
    if len(valid_images) < 3:
        return f"❌ Error: Need at least 3 photos. You have {len(valid_images)}. Please complete the capture process."
    
    name = name.strip().replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_messages = [f"--- Enrolling {name} with {CHOSEN_MODEL} ---"]
    log_messages.append(f"📸 Processing {len(valid_images)} captured photos...\n")
    known_embeddings = []
    
    for i, img in enumerate(valid_images):
        angle_name = CAPTURE_PROMPTS[i]["angle"] if i < len(CAPTURE_PROMPTS) else f"Photo {i+1}"
        try:
            # img is already a numpy array from webcam
            result = DeepFace.represent(
                img_path=img,
                model_name=CHOSEN_MODEL,
                enforce_detection=True,
                align=True
            )
            
            embedding = result[0]["embedding"]
            known_embeddings.append(embedding)
            log_messages.append(f"✔ {angle_name}: Face detected and processed")
            
        except ValueError:
            log_messages.append(f"⚠ {angle_name}: No face detected, skipping")
        except Exception as e:
            log_messages.append(f"⚠ {angle_name}: Error - {str(e)[:50]}")
    
    if len(known_embeddings) >= 2:
        master_embedding = np.mean(known_embeddings, axis=0)
        
        data = {
            "name": name,
            "model": CHOSEN_MODEL,
            "embedding": master_embedding,
            "image_count": len(known_embeddings)
        }
        
        output_path = os.path.join(OUTPUT_DIR, f"{name}_deepface.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        
        log_messages.append(f"\n{'='*40}")
        log_messages.append(f"✅ SUCCESS: {name} enrolled!")
        log_messages.append(f"📊 Embedding dimensions: {len(master_embedding)}")
        log_messages.append(f"🖼 Photos used: {len(known_embeddings)}/{len(valid_images)}")
        log_messages.append(f"📁 Saved to: {output_path}")
        
        return "\n".join(log_messages)
    else:
        log_messages.append(f"\n❌ FAILURE: Only {len(known_embeddings)} valid faces found.")
        log_messages.append("Please retake photos with better lighting and face visibility.")
        return "\n".join(log_messages)


def get_capture_instruction(step):
    """Get the instruction for current capture step."""
    if step < 0 or step >= len(CAPTURE_PROMPTS):
        return "✅ All photos captured!", "", "Complete"
    
    prompt = CAPTURE_PROMPTS[step]
    instruction = f"""
## 📸 Photo {step + 1} of 5: {prompt['angle']} View

# {prompt['icon']} {prompt['instruction']}

Position yourself and click **Capture** when ready.
"""
    return instruction, prompt['angle'], f"{step + 1}/5"


def list_enrolled_faces():
    """List all enrolled faces."""
    if not os.path.exists(OUTPUT_DIR):
        return "No enrolled faces found."

    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_deepface.pkl")]
    if not files:
        return "No enrolled faces found."

    enrolled = []
    for f in files:
        filepath = os.path.join(OUTPUT_DIR, f)
        try:
            with open(filepath, "rb") as file:
                data = pickle.load(file)
                img_count = data.get('image_count', 'N/A')
                enrolled.append(f"• {data['name']} (model: {data['model']}, images: {img_count})")
        except Exception as e:
            enrolled.append(f"• {f} (error reading)")

    return "📋 Enrolled Faces:\n" + "\n".join(enrolled)


def delete_enrolled_face(name):
    """Delete an enrolled face."""
    if not name or not name.strip():
        return "❌ Please enter a name to delete."
    
    name = name.strip().replace(" ", "_")
    filepath = os.path.join(OUTPUT_DIR, f"{name}_deepface.pkl")
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return f"✅ Deleted: {name}"
    else:
        return f"❌ Not found: {name}"


# --- Gradio Interface ---
with gr.Blocks(
    title="Facial Enrollment System",
    css="""
    .capture-instruction {
        font-size: 1.5em;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        margin: 10px 0;
    }
    .step-indicator {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
    }
    .captured-gallery img {
        border: 3px solid #4CAF50;
        border-radius: 8px;
    }
    """
) as demo:
    gr.Markdown("""
    # 🧑‍💻 Facial Enrollment System
    
    Enroll faces using guided camera capture with multiple angles for better recognition accuracy.
    """)
    
    # =========================================================================
    # TAB 1: Guided Camera Capture (Primary)
    # =========================================================================
    with gr.Tab("📸 Guided Camera Capture"):
        gr.Markdown("""
        ### Capture 5 photos at different angles for optimal enrollment
        
        The system will guide you through capturing photos from different angles 
        to create a robust facial profile.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Name input at top
                guided_name = gr.Textbox(
                    label="👤 Your Name",
                    placeholder="Enter your name first...",
                    info="This will be your identifier in the system"
                )
                
                # Current instruction display
                instruction_display = gr.Markdown(
                    value=get_capture_instruction(0)[0],
                    elem_classes=["capture-instruction"]
                )
                
                # Webcam for capture
                webcam = gr.Image(
                    label="Camera",
                    sources=["webcam"],
                    type="numpy",
                    height=400
                )
                
                # Progress indicator
                with gr.Row():
                    progress_text = gr.Markdown("**Progress: 0/5 photos captured**")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Captured Photos")
                
                # Store for captured images
                captured_1 = gr.Image(label="1. Front", height=100, interactive=False)
                captured_2 = gr.Image(label="2. Left", height=100, interactive=False)
                captured_3 = gr.Image(label="3. Right", height=100, interactive=False)
                captured_4 = gr.Image(label="4. Up", height=100, interactive=False)
                captured_5 = gr.Image(label="5. Down", height=100, interactive=False)
        
        # Hidden state for tracking
        current_step = gr.State(value=0)
        
        with gr.Row():
            capture_btn = gr.Button("📸 Capture Photo", variant="primary", size="lg")
            reset_btn = gr.Button("🔄 Reset All", variant="secondary")
        
        # Enrollment section
        gr.Markdown("---")
        with gr.Row():
            enroll_btn = gr.Button("✅ Complete Enrollment", variant="primary", size="lg")
        
        enrollment_result = gr.Textbox(
            label="Enrollment Result",
            lines=10,
            interactive=False
        )
        
        # Capture logic
        def capture_photo(current_image, step, img1, img2, img3, img4, img5):
            """Capture current webcam frame and advance to next step."""
            if current_image is None:
                return (
                    step, img1, img2, img3, img4, img5,
                    get_capture_instruction(step)[0],
                    f"**Progress: {step}/5 photos captured** ⚠️ No image detected!"
                )
            
            # Store image in appropriate slot
            images = [img1, img2, img3, img4, img5]
            if step < 5:
                images[step] = current_image
            
            # Advance step
            new_step = min(step + 1, 5)
            
            # Get new instruction
            if new_step >= 5:
                instruction = """
## ✅ All 5 Photos Captured!

Click **Complete Enrollment** below to process your photos and create your facial profile.
"""
                progress = "**Progress: 5/5 photos captured** ✅ Ready to enroll!"
            else:
                instruction = get_capture_instruction(new_step)[0]
                progress = f"**Progress: {new_step}/5 photos captured**"
            
            return (
                new_step,
                images[0], images[1], images[2], images[3], images[4],
                instruction,
                progress
            )
        
        def reset_capture():
            """Reset all captured images and start over."""
            return (
                0,  # step
                None, None, None, None, None,  # images
                get_capture_instruction(0)[0],  # instruction
                "**Progress: 0/5 photos captured**"  # progress
            )
        
        capture_btn.click(
            fn=capture_photo,
            inputs=[webcam, current_step, captured_1, captured_2, captured_3, captured_4, captured_5],
            outputs=[
                current_step,
                captured_1, captured_2, captured_3, captured_4, captured_5,
                instruction_display,
                progress_text
            ]
        )
        
        reset_btn.click(
            fn=reset_capture,
            outputs=[
                current_step,
                captured_1, captured_2, captured_3, captured_4, captured_5,
                instruction_display,
                progress_text
            ]
        )
        
        enroll_btn.click(
            fn=enroll_from_captured_images,
            inputs=[guided_name, captured_1, captured_2, captured_3, captured_4, captured_5],
            outputs=[enrollment_result]
        )
    
    # =========================================================================
    # TAB 2: Upload Images (Alternative)
    # =========================================================================
    with gr.Tab("📤 Upload Images"):
        gr.Markdown("""
        ### Alternative: Upload existing photos
        
        If you have photos already, upload them here instead of using the camera.
        """)
        
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(
                    label="Person's Name",
                    placeholder="Enter name (e.g., John Doe)"
                )
                image_input = gr.File(
                    label="Upload Face Images (3-5 recommended)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath"
                )
                enroll_upload_btn = gr.Button("🚀 Enroll Face", variant="primary")

            with gr.Column():
                output_log = gr.Textbox(
                    label="Enrollment Log",
                    lines=12,
                    interactive=False
                )
                output_file = gr.File(label="Downloaded Embedding File")

        enroll_upload_btn.click(
            fn=enroll_person_deepface,
            inputs=[name_input, image_input],
            outputs=[output_log, output_file]
        )
    
    # =========================================================================
    # TAB 3: Manage Enrolled Faces
    # =========================================================================
    with gr.Tab("📋 Manage Enrolled"):
        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
                enrolled_list = gr.Textbox(
                    label="Enrolled Faces",
                    lines=10,
                    interactive=False
                )
            
            with gr.Column():
                delete_name = gr.Textbox(
                    label="Name to Delete",
                    placeholder="Enter exact name..."
                )
                delete_btn = gr.Button("🗑️ Delete", variant="stop")
                delete_result = gr.Textbox(label="Result", lines=2, interactive=False)
        
        refresh_btn.click(fn=list_enrolled_faces, outputs=enrolled_list)
        delete_btn.click(fn=delete_enrolled_face, inputs=[delete_name], outputs=[delete_result])

    gr.Markdown("---")
    gr.Markdown(f"*Using DeepFace with {CHOSEN_MODEL} model • 5-angle capture for optimal accuracy*")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access on Jetson
        server_port=7861,
        share=False
    )
