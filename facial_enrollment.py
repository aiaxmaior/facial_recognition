from deepface import DeepFace
import numpy as np
import pickle
import gradio as gr
import os

# Configuration
# We use Facenet512 because it is widely considered the best balance of speed/accuracy
CHOSEN_MODEL = "Facenet512"
OUTPUT_DIR = "enrolled_faces"

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
            # 1. Detection & Extraction
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

    # 2. Averaging (Creating the Centroid)
    if len(known_embeddings) > 0:
        master_embedding = np.mean(known_embeddings, axis=0)

        # 3. Store
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
        with open(filepath, "rb") as file:
            data = pickle.load(file)
            enrolled.append(f"• {data['name']} (model: {data['model']})")

    return "📋 Enrolled Faces:\n" + "\n".join(enrolled)


# --- Gradio Interface ---
with gr.Blocks(title="Facial Enrollment System") as demo:
    gr.Markdown("# 🧑‍💻 Facial Enrollment System")
    gr.Markdown("Enroll faces using DeepFace with Facenet512 model for facial recognition.")

    with gr.Tab("📝 Enroll New Face"):
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(
                    label="Person's Name",
                    placeholder="Enter name (e.g., John Doe)",
                    info="This will be used as the identifier for the enrolled face."
                )
                image_input = gr.File(
                    label="Upload Face Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath"
                )
                enroll_btn = gr.Button("🚀 Enroll Face", variant="primary")

            with gr.Column():
                output_log = gr.Textbox(
                    label="Enrollment Log",
                    lines=12,
                    interactive=False
                )
                output_file = gr.File(label="Downloaded Embedding File")

        enroll_btn.click(
            fn=enroll_person_deepface,
            inputs=[name_input, image_input],
            outputs=[output_log, output_file]
        )

    with gr.Tab("📋 View Enrolled Faces"):
        refresh_btn = gr.Button("🔄 Refresh List")
        enrolled_list = gr.Textbox(
            label="Enrolled Faces",
            lines=10,
            interactive=False
        )
        refresh_btn.click(fn=list_enrolled_faces, outputs=enrolled_list)

    gr.Markdown("---")
    gr.Markdown("*Using DeepFace with Facenet512 model for optimal speed/accuracy balance.*")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(font=gr.themes.GoogleFont("TASA Orbiter")))