"""
Utility for loading known face encodings from disk.
"""

import logging
import os

import face_recognition
import numpy as np

from utils.config import KNOWN_FACES_DIR

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_known_faces(faces_dir: str = KNOWN_FACES_DIR) -> tuple[list[np.ndarray], list[str]]:
    """
    Load face encodings from the known-faces directory.

    The directory structure should be:
        known_faces/
            person_name/
                1.jpg
                2.jpg
                ...

    Args:
        faces_dir: Path to the root known-faces directory.

    Returns:
        A tuple of (encodings, names) where each encoding corresponds
        to the name at the same index.
    """
    encodings: list[np.ndarray] = []
    names: list[str] = []

    os.makedirs(faces_dir, exist_ok=True)

    person_dirs = [
        entry for entry in os.scandir(faces_dir)
        if entry.is_dir()
    ]

    if not person_dirs:
        logger.warning("No person folders found in '%s'.", faces_dir)
        return encodings, names

    for person_entry in sorted(person_dirs, key=lambda e: e.name):
        person_name = person_entry.name
        loaded = 0

        for image_entry in os.scandir(person_entry.path):
            ext = os.path.splitext(image_entry.name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            try:
                image = face_recognition.load_image_file(image_entry.path)
                face_encs = face_recognition.face_encodings(image)
            except Exception as exc:
                logger.warning("Could not process '%s': %s", image_entry.path, exc)
                continue

            if not face_encs:
                logger.warning("No face found in '%s', skipping.", image_entry.path)
                continue

            if len(face_encs) > 1:
                logger.warning(
                    "'%s' contains %d faces — skipping (corrupted registration image).",
                    image_entry.path, len(face_encs)
                )
                continue

            encodings.append(face_encs[0])
            names.append(person_name)
            loaded += 1

        if loaded:
            logger.info("Loaded %d encoding(s) for '%s'.", loaded, person_name)
        else:
            logger.warning("No usable images found for '%s'.", person_name)

    logger.info("Total encodings loaded: %d", len(encodings))
    return encodings, names
