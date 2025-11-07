# Hand Gesture Volume & Brightness Control (Python + Mediapipe)

This project allows you to control both your **system volume** and your **screen brightness** using simple hand gestures in front of your webcam.

✅ Thumb + Index = Volume Control  
✅ Index + Middle = Brightness Control  

Below you will find:
- Project description  
- Features  
- Installation  
- Full working code  
- Usage instructions  

Everything in ONE file.

---

## ✋ Features

### 🎚️ Volume Control
Thumb + Index finger distance controls system volume:

- Fingers **close** → lower volume  
- Fingers **far apart** → higher volume  

### 💡 Brightness Control
Index + Middle finger distance controls brightness:

- Fingers **close** → dimmer screen  
- Fingers **far apart** → brighter screen  

### ✅ Additional Features
- Real-time webcam tracking  
- Accurate hand landmark detection (Mediapipe)  
- Volume & brightness bars show feedback  
- Works smoothly on any Windows machine  

---

## 📦 Installation

Install all dependencies:

```bash
pip install opencv-python mediapipe numpy pycaw comtypes screen-brightness-control
