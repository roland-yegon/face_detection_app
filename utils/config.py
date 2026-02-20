"""
Configuration settings for the face detection application.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "data", "known_faces")

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Face Recognition ---
TOLERANCE = 0.48        # Lower = stricter matching (0.0–1.0)
FRAME_SCALE = 0.25      # Downscale factor for faster detection
MODEL = "hog"           # "hog" (CPU) or "cnn" (GPU, more accurate)

# --- Display ---
FONT = 0                # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_THICKNESS = 2
