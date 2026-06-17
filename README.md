# 🎮 Gesture Controls – Real-Time Hand Recognition & Device Control

<!-- ADD DEMO GIF HERE -->

**Real-time hand gesture recognition using MediaPipe + OpenCV. Recognize hand poses and translate them into device commands (mouse control, keyboard shortcuts, application automation).**

## The Problem

Traditional input methods (mouse, keyboard) are limiting for hands-free interaction. This project explores gesture-based device control using computer vision, enabling natural human-computer interaction for accessibility, gaming, and presentations.

## How It Works

1. **Capture Video**: Real-time webcam feed via OpenCV
2. **Detect Hands**: MediaPipe hand tracking (21 landmarks per hand)
3. **Recognize Gestures**: Match hand poses to predefined gesture library
4. **Map to Actions**: Translate gestures to keyboard/mouse/application commands
5. **Execute**: Control OS or applications without touching physical inputs

## Tech Stack

- **Python 3.7+**
- **MediaPipe** – hand pose estimation
- **OpenCV** – video capture & visualization
- **NumPy** – gesture math/angles
- **pyautogui (optional)** – mouse/keyboard control

## Quick Start

```bash
git clone https://github.com/georgepol1023/Gesture-Controls.git
cd Gesture-Controls
pip install -r requirements.txt
python gesture_control.py
```

Allow webcam access. Hold gestures in front of camera to trigger actions.

## Supported Gestures

| Gesture | Action | Use Case |
|---------|--------|----------|
| **Thumbs Up** | Volume Up | Media control |
| **Thumbs Down** | Volume Down | Media control |
| **Peace Sign** | Screenshot | Capture screen |
| **Open Palm** | Pause/Play | Media control |
| **Pinch (thumb+index)** | Click | Mouse replacement |
| **Fist** | Fullscreen Toggle | Presentation mode |

## Customization

Edit gesture mappings in `gesture_control.py`:

```python
gesture_actions = {
    'thumbs_up': lambda: pyautogui.press('volumeup'),
    'open_palm': lambda: pyautogui.press('playpause'),
    'pinch': lambda: pyautogui.click(),
}
```

## Output

- **Live Video**: Hand landmarks + recognized gesture label displayed on screen
- **Confidence Scores**: Real-time gesture matching confidence
- **Console Logs**: Triggered actions

## Key Features

✅ **Low-latency**: ~30ms recognition time  
✅ **Multi-hand**: Tracks both hands simultaneously  
✅ **Rotation-invariant**: Works at any hand angle  
✅ **Configurable**: Easy to add new gestures  

## Limitations

⚠️ **Environmental**:
- Requires good lighting for reliable detection
- Works best with minimal hand occlusion
- Hand size & distance from camera affects accuracy
- Background clutter can reduce performance

⚠️ **Technical**:
- No multi-gesture combinations yet
- Gesture hold duration is fixed
- No adaptive confidence thresholding

## Future Enhancements

- [ ] Gesture hold duration detection
- [ ] Combination gestures (two-hand sequences)
- [ ] ML-based custom gesture training
- [ ] Adaptive lighting normalization
- [ ] Integration with popular apps (PowerPoint, Spotify, OBS)
- [ ] Hand tracking metrics (speed, distance)

## Files

- `gesture_control.py` – Main gesture recognition engine
- `requirements.txt` – Dependencies

## Performance Tips

- Lower resolution if GPU is limited
- Reduce landmark update frequency for smoother tracking
- Use static image mode for pose estimation if video is jittery

## Author

[George Politis](https://github.com/georgepol1023)

## Resources

- [MediaPipe Hand Solutions](https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Hand Landmark Model](https://mediapipe.dev/)
