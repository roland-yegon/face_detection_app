# 🎯 Real-Time Face Recognition App

A Python application that detects and recognises faces from your webcam in real time using OpenCV and the `face_recognition` library.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

- 🟥 **Red box** — unknown face detected, not yet registered
- 🟩 **Green box** — face matched to a registered person
- 🔢 **Stable Unknown IDs** — one person always stays `Unknown 1`, never jumps to `Unknown 7`
- 🎯 **Best-match recognition** — uses `face_distance` to always pick the closest encoding, not just the first match
- ✂️ **Clean face crops** — saves only the cropped face with no overlays when registering
- ⚡ **Smooth video** — detection runs every 3 frames; labels draw on every frame

---

## 📁 Project Structure

```
face_detection_app/
├── app.py                  # Main entry point
├── clear_faces.py          # Wipe all saved face data
├── requirements.txt
├── utils/
│   ├── config.py           # All tunable settings
│   └── face_loader.py      # Loads face encodings from disk
└── data/
    └── known_faces/        # Auto-created on first run
        └── <name>/
            └── photo.jpg   # One cropped face photo per person
```

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/roland-yegon/face_detection_app.git
cd face-detection-app
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` requires `cmake` and a C++ compiler.
> On Ubuntu: `sudo apt install cmake build-essential`

### 4. Run the app

```bash
python3 app.py
```

---

## ⌨️ Controls

> The **OpenCV window must be focused** for keypresses to register.

| Key | Action |
|-----|--------|
| `s` | Save — register the unknown face currently in frame |
| `d` | Delete — remove a registered person by name |
| `r` | Reload — re-read all saved encodings from disk |
| `q` | Quit — close the camera and exit cleanly |

---

## 👤 Registering a New Person

1. Face the camera **alone** — your face should show a red `Unknown 1` box
2. Click the **Face Recognition** window to give it focus
3. Press **`s`**
4. Switch to the **terminal** and type the person's name, then press Enter
5. The box turns **green** once recognised ✅

> ⚠️ Only **Unknown** (red box) faces can be registered. If your box is already green, press `d` to delete the existing entry first.

---

## ⚙️ Configuration

All settings are in `utils/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TOLERANCE` | `0.48` | Recognition strictness. Lower = stricter (range 0–1) |
| `FRAME_SCALE` | `0.25` | Downscale factor for detection. Lower = faster |
| `MODEL` | `"hog"` | `"hog"` for CPU, `"cnn"` for GPU (more accurate) |
| `FRAME_WIDTH` | `640` | Camera resolution width |
| `FRAME_HEIGHT` | `480` | Camera resolution height |
| `CAMERA_INDEX` | `0` | Which camera to use. Try `1` if `0` doesn't work |

---

## 🛠️ Troubleshooting

**App loads old names even after deleting photos**

The folder structure persists even when photos are deleted. Run:
```bash
python3 clear_faces.py
```
Or find all `known_faces` directories and clear them:
```bash
find /home/$USER -type d -name "known_faces"
```

---

**Keys `s` / `d` / `r` / `q` don't respond**

Click the **Face Recognition** window to give it focus before pressing any key. After pressing `s`, switch to the terminal to type the name.

---

**Unknown counter keeps climbing**

The centroid tracker links faces across frames within 120px. If your face moves very fast, increase `MAX_DIST` in the `FaceTracker` class inside `app.py`.

---

**Everyone is identified as the same person**

Your saved photos likely contain multiple faces. Run `python3 clear_faces.py` and re-register each person individually with only their face in frame.

---

**Qt font warnings in terminal**

```
QFontDatabase: Cannot find font directory ...
```

These are harmless warnings from OpenCV's Qt backend. The app works fine — safely ignore them.

---

## 📦 Requirements

```
opencv-python>=4.8
face-recognition>=1.3
dlib>=19.24
numpy>=1.24
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.