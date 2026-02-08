# Face Detection App

Real-time face detection using Python and OpenCV. This small app uses a Haar Cascade classifier to detect faces from your webcam and draws a green rectangle around each detected face.

---

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Description
This repository contains a minimal example of real-time face detection using OpenCV's Haar Cascade classifier. The main script is [app.py](app.py) and the Haar cascade file is included as [haarcascade_frontalface_default.xml](haarcascade_frontalface_default.xml).

---

## Requirements
- **Python 3.8+**
- System camera (webcam) access
- Python package: see [requirements.txt](requirements.txt)

---

## Setup
1. Clone the repository (if you haven't already):

```bash
git clone https://github.com/roland-yegon/face_detection_app.git
cd face_detection_app
```

2. (Recommended) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage
- Run the app:

```bash
python app.py
```

- Controls: **Press `q`** in the window to quit the application.
- The app opens your default camera (device `0`). To use a different camera, modify the `cv2.VideoCapture(0)` line in [app.py](app.py).

---

## Troubleshooting
- **No camera detected:** Ensure the camera is connected and not used by another application. Confirm `/dev/video0` (Linux) exists and is accessible.
- **Permission errors on Linux:** You may need to run with appropriate permissions or adjust udev rules for your camera device.
- **Black window or no frames:** Try changing the camera index in [app.py](app.py) or test the camera with another app (e.g., `cheese`).
- **Cascade file not found:** Ensure [haarcascade_frontalface_default.xml](haarcascade_frontalface_default.xml) is present in the project root.

---

## Contributing
Contributions and improvements are welcome. Open an issue or a pull request with suggested changes.

---

## License
This project is licensed under the terms in [LICENCE.md](LICENCE.md).
