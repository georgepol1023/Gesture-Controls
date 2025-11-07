import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# -----------------------------
# MEDIAPIPE HANDS SETUP
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    img_h, img_w, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handType in zip(results.multi_hand_landmarks, results.multi_handedness):
            lm_list = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in handLms.landmark]
            handedness = handType.classification[0].label  # 'Left' or 'Right'

            if handedness == "Right":
                # Actually on screen: left hand
                # Hand mode
                center_y = int(np.mean([p[1] for p in lm_list]))
                brightness = int((1 - (center_y / img_h)) * 100)
                cv2.putText(img, "Left Hand Mode", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif handedness == "Left":
                # Actually on screen: right hand
                # Two-finger mode
                tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                fingers_up = []
                for tip in tips_ids:
                    if lm_list[tip][1] < lm_list[tip - 2][1]:
                        fingers_up.append(tip)

                if len(fingers_up) >= 2:
                    f1, f2 = fingers_up[:2]
                    x1, y1 = lm_list[f1]
                    x2, y2 = lm_list[f2]
                    finger_distance = np.hypot(x2 - x1, y2 - y1)
                    brightness = int(np.interp(finger_distance, [20, 200], [0, 100]))
                    cv2.putText(img, "Right Hand Mode (Fingers)", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    continue

            else:
                continue

            # Set brightness
            sbc.set_brightness(brightness)

            # Draw hand landmarks
            for x, y in lm_list:
                cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Brightness bar
            bar_pos = np.interp(brightness, [0, 100], [400, 100])
            cv2.rectangle(img, (100, 100), (135, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (100, int(bar_pos)), (135, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"Brightness: {brightness}%", (80, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show webcam
    cv2.imshow("Hand Brightness Control", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
