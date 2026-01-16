import argparse
import os
import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

class FaceAuthSystem:
    def __init__(self, db_folder="enrolled_faces", model="Facenet512", threshold=0.30):
        self.db_folder = db_folder
        self.model_name = model
        self.threshold = threshold
        
        # Ensure DB folder exists
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)

    def enroll(self, name, image_paths):
        """
        Reads images, generates an averaged embedding, and saves to disk.
        """
        print(f"--- Enrolling {name} ---")
        embeddings = []
        
        for p in image_paths:
            if not os.path.exists(p):
                print(f"⚠ File not found: {p}")
                continue
                
            try:
                # Get embedding (force detection to ensure quality)
                res = DeepFace.represent(img_path=p, model_name=self.model_name, enforce_detection=True, align=True)
                embeddings.append(res[0]["embedding"])
                print(f"✔ Processed: {p}")
            except Exception as e:
                print(f"⚠ Skipping {p}: {e}")

        if not embeddings:
            print("❌ Failure: No valid faces processed.")
            return

        # Create 'Master' Vector (Centroid)
        master_vector = np.mean(embeddings, axis=0)
        
        # Save
        save_path = os.path.join(self.db_folder, f"{name}_deepface.pkl")
        data = {
            "name": name,
            "model": self.model_name,
            "embedding": master_vector
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
            
        print(f"✅ Success: Saved {name} to {save_path}")

    def load_database(self):
        """Loads all .pkl files into memory."""
        db = {}
        for fname in os.listdir(self.db_folder):
            if fname.endswith("_deepface.pkl"):
                path = os.path.join(self.db_folder, fname)
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                        if data["model"] == self.model_name:
                            db[data["name"]] = data["embedding"]
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        return db

    def match(self, target_image_path):
        """
        Compares target image against the loaded database.
        """
        print(f"--- Matching: {target_image_path} ---")
        
        # 1. Load DB (In production, load this once at startup, not every request)
        database = self.load_database()
        if not database:
            print("⚠ Database is empty. Please enroll someone first.")
            return

        # 2. Get Target Vector
        try:
            res = DeepFace.represent(img_path=target_image_path, model_name=self.model_name, enforce_detection=True, align=True)
            target_vector = res[0]["embedding"]
        except Exception as e:
            print(f"❌ Error processing target image: {e}")
            return

        # 3. Compare
        best_match = "Unknown"
        best_score = 1.0 # 1.0 is max distance (no match)

        for name, db_vector in database.items():
            score = cosine(target_vector, db_vector)
            if score < best_score:
                best_score = score
                best_match = name

        # 4. Result
        if best_score <= self.threshold:
            print(f"✅ MATCH FOUND: {best_match}")
            print(f"📊 Distance: {best_score:.4f} (Threshold: {self.threshold})")
        else:
            print(f"⛔ UNKNOWN PERSON")
            print(f"📊 Best Guess: {best_match} at distance {best_score:.4f} (Too far)")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Auth System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: Enroll
    # Usage: python face_system.py enroll -n "John Doe" -i photo1.jpg photo2.jpg
    parser_enroll = subparsers.add_parser("enroll", help="Enroll a new user")
    parser_enroll.add_argument("-n", "--name", required=True, help="Name of the person")
    parser_enroll.add_argument("-i", "--images", nargs="+", required=True, help="List of image paths")

    # Sub-command: Match
    # Usage: python face_system.py match -t target_photo.jpg
    parser_match = subparsers.add_parser("match", help="Identify a person from an image")
    parser_match.add_argument("-t", "--target", required=True, help="Path to the image to check")

    args = parser.parse_args()
    
    # Initialize System
    system = FaceAuthSystem()

    if args.command == "enroll":
        system.enroll(args.name, args.images)
    elif args.command == "match":
        system.match(args.target)