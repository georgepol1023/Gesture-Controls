import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# -----------------------------
# MEDIAPIPE HANDS SETUP
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# WINDOWS VOLUME SETUP
# -----------------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol, _ = vol_range  # dB values

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            lm_list = []

            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # -----------------------------
            # VOLUME GESTURE: Thumb (4) + Index (8)
            # -----------------------------
            x1, y1 = lm_list[4]   # Thumb tip
            x2, y2 = lm_list[8]   # Index tip

            cv2.circle(img, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            length_vol = int(np.hypot(x2 - x1, y2 - y1))

            # Map distance to volume range
            vol = np.interp(length_vol, [20, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Volume bar (display)
            vol_bar = np.interp(length_vol, [20, 200], [400, 100])
            cv2.rectangle(img, (50, 100), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, "Volume", (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            # -----------------------------
            # BRIGHTNESS GESTURE: Index (8) + Middle (12)
            # -----------------------------
            x3, y3 = lm_list[12]  # Middle finger tip

            cv2.circle(img, (x3, y3), 8, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 2)

            length_brightness = int(np.hypot(x3 - x2, y3 - y2))

            # Map distance to brightness 0 → 100%
            brightness = int(np.interp(length_brightness, [20, 200], [0, 100]))
            sbc.set_brightness(brightness)

            # Brightness bar
            bright_bar = np.interp(length_brightness, [20, 200], [400, 100])
            cv2.rectangle(img, (100, 100), (135, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (100, int(bright_bar)), (135, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Brightness", (80, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show image
    cv2.imshow("Gesture Control (Volume + Brightness)", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()