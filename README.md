# Hand Gesture Brightness Control

Control your screen brightness using hand gestures with your webcam! This project uses **MediaPipe** for hand detection and **screen_brightness_control** to adjust brightness on Windows.

---

## Features

- **Right hand (on screen left)**: Control brightness by pinching or spreading **two fingers**.  
- **Left hand (on screen right)**: Control brightness by moving your **fist up and down**.  
- Supports **simultaneous use** of both hands.  
- Visual feedback with **landmarks** and **brightness bar**.  
- Camera flip is accounted for.

---

## Requirements

- Python 3.10+  
- Libraries:

```bash
pip install opencv-python mediapipe numpy screen_brightness_control
