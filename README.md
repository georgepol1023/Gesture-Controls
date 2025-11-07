# Hand Gesture Brightness Control

This project allows you to control your computer screen brightness using **hand gestures** detected through your webcam. Specifically, it uses the distance between your **index** and **middle fingers** to adjust the brightness in real time.

## Features

- Detects a single hand using **MediaPipe Hands**.
- Controls screen brightness with **index + middle finger distance**.
- Displays a **brightness bar** on the webcam feed.
- Real-time visualization of hand landmarks and gesture.

## Requirements

- Python 3.10+
- Windows 10 or 11
- Webcam

### Python Packages

Install the required packages:

```bash
pip install opencv-python mediapipe numpy screen_brightness_control
