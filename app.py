"""
Real-time face recognition app using OpenCV and face_recognition.

Controls:
  s — Save current frame and register a new person
  r — Reload known faces from disk
  q — Quit
"""

import logging
import os
import sys

import cv2
import face_recognition
import numpy as np

from utils.config import (
    BOX_THICKNESS, CAMERA_INDEX, FONT, FONT_SCALE,
    FONT_THICKNESS, FRAME_HEIGHT, FRAME_SCALE, FRAME_WIDTH,
    KNOWN_FACES_DIR, MODEL, TOLERANCE,
)
from utils.face_loader import load_known_faces

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLOR_UNKNOWN = (0, 0, 255)   # Red   — unrecognised
COLOR_KNOWN   = (0, 255, 0)   # Green — recognised

RECOGNITION_INTERVAL = 3  # Run detection every N frames for smooth video


# ---------------------------------------------------------------------------
# Centroid-distance tracker
# ---------------------------------------------------------------------------

def _centroid(box):
    top, right, bottom, left = box
    return ((left + right) // 2, (top + bottom) // 2)


def _distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


class FaceTracker:
    """
    Tracks faces across frames by matching centroids.
    - Keeps Unknown IDs compact (1…N, no gaps, no runaway counter).
    - Promotes a track to a real name the moment it is recognised.
    """

    # A detection is linked to an existing track if their centroids are
    # within this many pixels of each other.
    MAX_DIST   = 120   # px — tune up if faces jump around a lot
    MAX_MISSED = 20    # frames before a vanished track is removed

    def __init__(self):
        self._tracks = {}   # tid → {cx, cy, name, missed}
        self._next_tid = 0

    # ------------------------------------------------------------------ #
    def update(self, detections: list) -> list:
        """
        Args:
            detections: list of (box, name) where box=(top,right,bottom,left)
                        and name is a recognised string or "" for unknown.
        Returns:
            list of (box, display_label)
        """
        # Mark all tracks as unmatched
        for t in self._tracks.values():
            t["matched"] = False

        results = []   # (box, tid) — we'll resolve labels after renumbering

        for box, name in detections:
            cx, cy = _centroid(box)

            # Find the nearest unmatched track within MAX_DIST
            best_tid, best_dist = None, self.MAX_DIST
            for tid, track in self._tracks.items():
                if track["matched"]:
                    continue
                d = _distance((cx, cy), (track["cx"], track["cy"]))
                if d < best_dist:
                    best_dist, best_tid = d, tid

            if best_tid is not None:
                t = self._tracks[best_tid]
                t["cx"]      = cx
                t["cy"]      = cy
                t["missed"]  = 0
                t["matched"] = True
                if name:                 # promote to real name once recognised
                    t["name"] = name
                results.append((box, best_tid))
            else:
                # Brand-new face
                tid = self._next_tid
                self._next_tid += 1
                self._tracks[tid] = {
                    "cx": cx, "cy": cy,
                    "name": name if name else "_new_",
                    "missed": 0, "matched": True,
                }
                results.append((box, tid))

        # Age out stale tracks
        for tid in [tid for tid, t in self._tracks.items() if not t["matched"]]:
            self._tracks[tid]["missed"] += 1
            if self._tracks[tid]["missed"] > self.MAX_MISSED:
                del self._tracks[tid]

        # Renumber unknowns compactly: Unknown 1, 2, … — no gaps
        unknown_tids = sorted(
            tid for tid, t in self._tracks.items()
            if t["name"].startswith("Unknown") or t["name"] == "_new_"
        )
        for i, tid in enumerate(unknown_tids, start=1):
            self._tracks[tid]["name"] = f"Unknown {i}"

        # Build final output using the now-stable labels
        return [(box, self._tracks[tid]["name"]) for box, tid in results]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Cannot open camera at index %d.", index)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def scale_locations(locs_small, scale):
    factor = int(round(1.0 / scale))
    return [(t * factor, r * factor, b * factor, l * factor)
            for t, r, b, l in locs_small]


def draw_face(frame, top, right, bottom, left, name):
    color = COLOR_KNOWN if not name.startswith("Unknown") else COLOR_UNKNOWN
    cv2.rectangle(frame, (left, top), (right, bottom), color, BOX_THICKNESS)
    cv2.putText(frame, name, (left, top - 10),
                FONT, FONT_SCALE, color, FONT_THICKNESS)


def save_face(clean_frame, last_results):
    """
    Save a cropped face from clean_frame (no drawings).
    Only saves an Unknown face — refuses to save an already-recognised face.
    If multiple unknowns, saves the first one found.
    """
    person_name = input("Enter person's name (blank to cancel): ").strip().lower()
    if not person_name:
        logger.info("Save cancelled.")
        return None

    if not last_results:
        logger.warning("No face detected in frame — move closer and try again.")
        return None

    # Find unknown faces only
    unknowns = [(box, name) for box, name in last_results if name.startswith("Unknown")]
    if not unknowns:
        logger.warning("No unknown face in frame — only unrecognised faces can be registered.")
        return None

    # Use the first unknown face detected
    (top, right, bottom, left), _ = unknowns[0]

    # Add a small padding around the face
    h, w = clean_frame.shape[:2]
    pad = 20
    top    = max(0, top    - pad)
    left   = max(0, left   - pad)
    bottom = min(h, bottom + pad)
    right  = min(w, right  + pad)

    face_crop = clean_frame[top:bottom, left:right]

    folder = os.path.join(KNOWN_FACES_DIR, person_name)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, "photo.jpg")

    if cv2.imwrite(path, face_crop):
        logger.info("Saved '%s' → %s", person_name, path)
        return load_known_faces()

    logger.error("Could not write to %s", path)
    return None


def delete_person(known_faces_dir: str) -> tuple[list, list] | None:
    """Prompt for a name and delete that person from known faces."""
    person_name = input("Enter name to delete (blank to cancel): ").strip().lower()
    if not person_name:
        logger.info("Delete cancelled.")
        return None

    folder = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(folder):
        logger.warning("No saved face found for '%s'.", person_name)
        return None

    import shutil
    shutil.rmtree(folder)
    logger.info("Deleted '%s'.", person_name)
    return load_known_faces()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting face recognition system…")
    known_encodings, known_names = load_known_faces()

    video   = open_camera()
    tracker = FaceTracker()

    last_results = []
    frame_count  = 0

    logger.info("Controls:  s = save face  |  d = delete person  |  r = reload faces  |  q = quit")

    while True:
        ret, frame = video.read()
        if not ret:
            logger.error("Camera read failed.")
            break

        frame_count += 1

        # ---- Recognition (every N frames) ----
        if frame_count % RECOGNITION_INTERVAL == 0:
            small_rgb = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE,
            )

            locs_small = face_recognition.face_locations(small_rgb, model=MODEL)
            encodings  = face_recognition.face_encodings(small_rgb, locs_small)
            locations  = scale_locations(locs_small, FRAME_SCALE)

            detections = []
            for enc, loc in zip(encodings, locations):
                name = ""
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, enc)
                    best_idx  = int(np.argmin(distances))
                    if distances[best_idx] < TOLERANCE:
                        name = known_names[best_idx]
                detections.append((loc, name))

            last_results = tracker.update(detections)

        # Keep a clean copy before drawing so saved photos have no overlays
        clean_frame = frame.copy()

        # ---- Draw on every frame ----
        for (top, right, bottom, left), name in last_results:
            draw_face(frame, top, right, bottom, left, name)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            result = save_face(clean_frame, last_results)
            if result:
                known_encodings, known_names = result

        elif key == ord("d"):
            result = delete_person(KNOWN_FACES_DIR)
            if result:
                known_encodings, known_names = result

        elif key == ord("r"):
            logger.info("Reloading faces…")
            known_encodings, known_names = load_known_faces()

        elif key == ord("q"):
            logger.info("Quitting.")
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
