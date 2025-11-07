import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# -----------------------------
# MEDIAPIPE HANDS SETUP
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

            # Safety: need at least index (8) and middle (12) tips
            if len(lm_list) < 13:
                continue

            # -----------------------------
            # BRIGHTNESS GESTURE: Index (8) + Middle (12)
            # -----------------------------
            x_index, y_index = lm_list[8]
            x_middle, y_middle = lm_list[12]

            cv2.circle(img, (x_index, y_index), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x_middle, y_middle), 8, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x_index, y_index), (x_middle, y_middle), (0, 255, 0), 2)

            length_brightness = int(np.hypot(x_middle - x_index, y_middle - y_index))
            brightness = int(np.interp(length_brightness, [20, 200], [0, 100]))
            sbc.set_brightness(brightness)

            # Brightness bar display
            bright_bar = np.interp(length_brightness, [20, 200], [400, 100])
            cv2.rectangle(img, (100, 100), (135, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (100, int(bright_bar)), (135, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Brightness", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show webcam
    cv2.imshow("Brightness Control", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
