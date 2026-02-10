import cv2
import face_recognition
import os
from utils.config import *
from utils.face_loader import load_known_faces

print("[INFO] Starting System...")

known_encodings, known_names = load_known_faces()

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

unknown_counter = 0
unknown_tracks = {}

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_rgb = cv2.resize(rgb, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

    locations_small = face_recognition.face_locations(small_rgb, model=MODEL)
    encodings = face_recognition.face_encodings(small_rgb, locations_small)

    locations = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in locations_small]

    current_unknowns = []

    for encoding, (top, right, bottom, left) in zip(encodings, locations):

        matches = face_recognition.compare_faces(known_encodings, encoding, TOLERANCE)
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
        else:
            # Assign stable Unknown IDs
            key = (top//50, left//50)
            if key not in unknown_tracks:
                unknown_counter += 1
                unknown_tracks[key] = f"Unknown {unknown_counter}"
            name = unknown_tracks[key]

        current_unknowns.append(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # 📸 SAVE FACE (multiple images per person)
    if key == ord("s"):
        person_name = input("Enter person's name: ").lower()
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        count = len(os.listdir(person_folder)) + 1
        path = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(path, frame)

        print(f"[INFO] Saved image {count} for {person_name}")
        known_encodings, known_names = load_known_faces()

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
