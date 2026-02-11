import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark

                # Finger tips
                thumb_tip = lm[4]
                index_tip = lm[8]
                middle_tip = lm[12]
                ring_tip = lm[16]
                pinky_tip = lm[20]

                # Finger PIP joints
                thumb_ip = lm[3]
                index_pip = lm[6]
                middle_pip = lm[10]
                ring_pip = lm[14]
                pinky_pip = lm[18]

                # ------------------
                # THUMBS UP
                # ------------------
                if (thumb_tip.y < thumb_ip.y and
                    index_tip.y > index_pip.y and
                    middle_tip.y > middle_pip.y and
                    ring_tip.y > ring_pip.y and
                    pinky_tip.y > pinky_pip.y):

                    cv2.putText(frame, "Thumbs Up 👍", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # ------------------
                # PEACE SIGN ✌
                # ------------------
                elif (index_tip.y < index_pip.y and
                      middle_tip.y < middle_pip.y and
                      ring_tip.y > ring_pip.y and
                      pinky_tip.y > pinky_pip.y):

                    cv2.putText(frame, "Peace ✌", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                # ------------------
                # ROCK SIGN 🤟
                # ------------------
                elif (index_tip.y < index_pip.y and
                      middle_tip.y > middle_pip.y and
                      ring_tip.y > ring_pip.y and
                      pinky_tip.y < pinky_pip.y):

                    cv2.putText(frame, "Rock Sign 🤟", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
