import os
import face_recognition
from utils.config import KNOWN_FACES_DIR

def load_known_faces():
    encodings = []
    names = []

    print("[INFO] Loading known faces...")

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):
            path = os.path.join(person_folder, image_name)

            image = face_recognition.load_image_file(path)
            face_enc = face_recognition.face_encodings(image)

            if face_enc:
                encodings.append(face_enc[0])
                names.append(person_name)

    print(f"[INFO] Loaded {len(encodings)} encodings")
    return encodings, names
