import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "data", "known_faces")

TOLERANCE = 0.48
FRAME_SCALE = 0.25
MODEL = "hog"
