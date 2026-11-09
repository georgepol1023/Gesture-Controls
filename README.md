# Gesture Controls

Control your screen brightness with your hands, using a webcam, MediaPipe hand tracking and OpenCV.

## How it works

MediaPipe finds 21 landmarks on each hand (up to two hands) in every webcam frame. The script then
has two modes, depending on which hand it sees:

- **Left hand: height mode.** Raise your hand to brighten the screen and lower it to dim it.
  Brightness follows the average height of the hand's landmarks in the frame.
- **Right hand: finger mode.** Raise two fingers and move them apart or together. The distance
  between the two fingertips (20 to 200 pixels) maps to 0 to 100% brightness, shown as a red line
  between them.

The brightness is set with [`screen_brightness_control`](https://github.com/Crozzers/screen_brightness_control),
and a bar on screen shows the current level. Press `q` to quit.

## Run it

```bash
git clone https://github.com/georgepol1023/Gesture-Controls.git
cd Gesture-Controls
pip install -r requirements.txt
python gesture_controls.py
```

Allow webcam access when asked. Brightness control works on laptop screens and on monitors that
support it (DDC/CI); on some external monitors it has no effect.

## Limitations

- Needs reasonable lighting so MediaPipe can track the hand reliably.
- MediaPipe labels hands as seen in the mirrored image, so "left" and "right" are from the camera's
  point of view. The script accounts for this.
- Finger mode uses the first two raised fingers it finds, so it works best with only two fingers up.

## Built with

Python, OpenCV, MediaPipe, NumPy and `screen_brightness_control`.
