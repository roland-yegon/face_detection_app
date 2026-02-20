"""
Run this once to wipe all saved face data so you can start fresh.
    python clear_faces.py
"""
import shutil
import os
from utils.config import KNOWN_FACES_DIR

if os.path.exists(KNOWN_FACES_DIR):
    shutil.rmtree(KNOWN_FACES_DIR)
    os.makedirs(KNOWN_FACES_DIR)
    print(f"[INFO] Cleared all saved faces in '{KNOWN_FACES_DIR}'. Start fresh!")
else:
    print("[INFO] Nothing to clear.")
