"""
Python API wrapper for DeepFace enrollment processing.
This provides an HTTP interface to the existing enrollment logic.
"""

import os
import sys
import base64
import logging
from io import BytesIO
from typing import List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DeepFace and existing enrollment logic
try:
    from deepface import DeepFace
    import cv2
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available - running in mock mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (matching facial_enrollment.py)
CHOSEN_MODEL = "ArcFace"
CHOSEN_DETECTOR = "yolov8"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../enrolled_faces")

app = Flask(__name__)
CORS(app)


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode a base64 image string to numpy array."""
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_str)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    return img


def save_face_embedding_to_db(
    db_path: str,
    name: str,
    model: str,
    detector: str,
    embedding: np.ndarray,
    embedding_normalized: np.ndarray,
    image_count: int,
    **kwargs
) -> bool:
    """Save face embedding to SQLite database."""
    import sqlite3
    import datetime
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            first_name TEXT,
            last_name TEXT,
            employee_id TEXT,
            email TEXT,
            model TEXT NOT NULL,
            detector TEXT,
            embedding BLOB NOT NULL,
            embedding_normalized BLOB,
            image_count INTEGER,
            profile_image_path TEXT,
            images_directory TEXT,
            enrolled_at TEXT
        )
    """)
    
    # Convert numpy arrays to bytes
    embedding_bytes = embedding.astype(np.float32).tobytes()
    normalized_bytes = embedding_normalized.astype(np.float32).tobytes() if embedding_normalized is not None else None
    
    # Insert or replace
    cursor.execute("""
        INSERT OR REPLACE INTO faces 
        (name, first_name, last_name, employee_id, email, model, detector, 
         embedding, embedding_normalized, image_count, profile_image_path, images_directory, enrolled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        kwargs.get('first_name'),
        kwargs.get('last_name'),
        kwargs.get('employee_id'),
        kwargs.get('email'),
        model,
        detector,
        embedding_bytes,
        normalized_bytes,
        image_count,
        kwargs.get('profile_image_path'),
        kwargs.get('images_directory'),
        datetime.datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    return True


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'deepface_available': DEEPFACE_AVAILABLE,
        'model': CHOSEN_MODEL,
        'detector': CHOSEN_DETECTOR,
    })


@app.route('/api/process', methods=['POST'])
def process_enrollment():
    """
    Process captured images and generate face embeddings.
    
    Request body:
    {
        "user_id": "string",
        "captures": [
            {"pose": "front", "image_data": "base64..."},
            ...
        ]
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        captures = data.get('captures', [])
        
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        if len(captures) < 5:
            return jsonify({'success': False, 'error': 'At least 5 captures required'}), 400
        
        logger.info(f"Processing enrollment for user: {user_id}")
        logger.info(f"Received {len(captures)} captures")
        
        if not DEEPFACE_AVAILABLE:
            # Mock mode for testing without DeepFace
            logger.warning("Running in mock mode - no actual embedding generation")
            return jsonify({
                'success': True,
                'embedding_count': len(captures),
                'profile_image_path': f'/images/{user_id}/profile_128.jpg',
                'mock_mode': True,
            })
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        images_dir = os.path.join(OUTPUT_DIR, "images", user_id)
        os.makedirs(images_dir, exist_ok=True)
        
        embeddings = []
        pose_names = ['front', 'left', 'right', 'up', 'down']
        profile_image_path = os.path.join(images_dir, "profile_128.jpg")
        
        for i, capture in enumerate(captures):
            pose = capture.get('pose', pose_names[i] if i < len(pose_names) else f'pose_{i}')
            image_data = capture.get('image_data', '')
            
            try:
                # Decode image
                img = decode_base64_image(image_data)
                logger.info(f"Processing {pose}: shape={img.shape}")
                
                # Save image
                image_path = os.path.join(images_dir, f"{pose}.jpg")
                cv2.imwrite(image_path, img)
                
                # Create profile thumbnail from front image
                if i == 0:
                    try:
                        faces = DeepFace.extract_faces(
                            img_path=img,
                            detector_backend=CHOSEN_DETECTOR,
                            enforce_detection=True,
                            align=True
                        )
                        if faces and len(faces) > 0:
                            face_img = faces[0]['face']
                            face_img = (face_img * 255).astype(np.uint8)
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                            thumbnail = cv2.resize(face_img, (128, 128))
                        else:
                            thumbnail = cv2.resize(img, (128, 128))
                    except Exception as e:
                        logger.warning(f"Face extraction failed: {e}")
                        h, w = img.shape[:2]
                        size = min(h, w)
                        y_start = (h - size) // 2
                        x_start = (w - size) // 2
                        cropped = img[y_start:y_start+size, x_start:x_start+size]
                        thumbnail = cv2.resize(cropped, (128, 128))
                    
                    cv2.imwrite(profile_image_path, thumbnail)
                    logger.info(f"Created profile thumbnail: {profile_image_path}")
                
                # Generate embedding
                try:
                    result = DeepFace.represent(
                        img_path=img,
                        model_name=CHOSEN_MODEL,
                        detector_backend=CHOSEN_DETECTOR,
                        enforce_detection=True,
                        align=True
                    )
                except ValueError:
                    logger.warning(f"Detection failed for {pose}, retrying with skip")
                    result = DeepFace.represent(
                        img_path=img,
                        model_name=CHOSEN_MODEL,
                        detector_backend="skip",
                        enforce_detection=False,
                        align=False
                    )
                
                embeddings.append(result[0]["embedding"])
                logger.info(f"Generated embedding for {pose}")
                
            except Exception as e:
                logger.error(f"Failed to process {pose}: {e}")
        
        if len(embeddings) < 2:
            return jsonify({
                'success': False,
                'error': f'Only {len(embeddings)} valid faces detected',
            }), 400
        
        # Create master embedding (average of all)
        master_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(master_embedding)
        normalized_embedding = master_embedding / norm if norm > 0 else master_embedding
        
        # Save to database
        db_path = os.path.join(OUTPUT_DIR, "faces.db")
        save_face_embedding_to_db(
            db_path=db_path,
            name=user_id,
            model=CHOSEN_MODEL,
            detector=CHOSEN_DETECTOR,
            embedding=master_embedding,
            embedding_normalized=normalized_embedding,
            image_count=len(embeddings),
            employee_id=user_id,
            profile_image_path=profile_image_path,
            images_directory=images_dir,
        )
        
        logger.info(f"Enrollment saved to: {db_path}")
        
        return jsonify({
            'success': True,
            'embedding_count': len(embeddings),
            'profile_image_path': profile_image_path,
        })
        
    except Exception as e:
        logger.exception("Processing error")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/embedding/<user_id>', methods=['GET'])
def get_embedding(user_id: str):
    """Get embedding for a user."""
    import sqlite3
    
    db_path = os.path.join(OUTPUT_DIR, "faces.db")
    
    if not os.path.exists(db_path):
        return jsonify({'error': 'Not found'}), 404
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM faces WHERE name = ? OR employee_id = ?", (user_id, user_id))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'error': 'Not found'}), 404
    
    embedding = np.frombuffer(row['embedding'], dtype=np.float32)
    
    return jsonify({
        'user_id': user_id,
        'embedding': embedding.tolist(),
        'model': row['model'],
        'enrolled_at': row['enrolled_at'],
    })


@app.route('/api/enrollment/<user_id>', methods=['DELETE'])
def delete_enrollment(user_id: str):
    """Delete enrollment for a user."""
    import sqlite3
    import shutil
    
    db_path = os.path.join(OUTPUT_DIR, "faces.db")
    
    if not os.path.exists(db_path):
        return jsonify({'error': 'Not found'}), 404
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get images directory before deleting
    cursor.execute("SELECT images_directory FROM faces WHERE name = ? OR employee_id = ?", (user_id, user_id))
    row = cursor.fetchone()
    
    if row and row[0] and os.path.exists(row[0]):
        shutil.rmtree(row[0])
    
    cursor.execute("DELETE FROM faces WHERE name = ? OR employee_id = ?", (user_id, user_id))
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    
    if not deleted:
        return jsonify({'error': 'Not found'}), 404
    
    return jsonify({'success': True, 'message': 'Enrollment deleted'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"🐍 Python Enrollment API starting on http://localhost:{port}")
    print(f"   DeepFace available: {DEEPFACE_AVAILABLE}")
    print(f"   Model: {CHOSEN_MODEL}")
    print(f"   Detector: {CHOSEN_DETECTOR}")
    print(f"   Output dir: {OUTPUT_DIR}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
