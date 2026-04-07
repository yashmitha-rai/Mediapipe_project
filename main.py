import cv2
import mediapipe as mp

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands, \
     mp_face_detection.FaceDetection(
        min_detection_confidence=0.7) as face_det, \
     mp_face_mesh.FaceMesh(
        refine_landmarks=True) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process all models
        hand_result = hands.process(rgb)
        face_result = face_det.process(rgb)
        mesh_result = face_mesh.process(rgb)

        # ---------------- FACE DETECTION ----------------
        if face_result.detections:
            cv2.putText(frame, "Person Detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            for detection in face_result.detections:
                mp_drawing.draw_detection(frame, detection)

        # ---------------- BLINK + YAWN ----------------
        if mesh_result.multi_face_landmarks:
            for face_landmarks in mesh_result.multi_face_landmarks:

                lm = face_landmarks.landmark

                # Blink Detection
                left_eye = abs(lm[159].y - lm[145].y)
                right_eye = abs(lm[386].y - lm[374].y)

                if left_eye < 0.01 and right_eye < 0.01:
                    cv2.putText(frame, "Blink Detected",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2)

                # Yawn Detection
                mouth_open = abs(lm[13].y - lm[14].y)

                if mouth_open > 0.05:
                    cv2.putText(frame, "Yawn Detected",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2)

        # ---------------- HAND DETECTION ----------------
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark

                # ---------------- HAND RAISED ----------------
                if lm[0].y < 0.5:
                    cv2.putText(frame, "Student wants to speak",
                                (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

                # ---------------- FINGER COUNT ----------------
                finger_count = 0
                tips = [8, 12, 16, 20]

                for tip in tips:
                    if lm[tip].y < lm[tip - 2].y:
                        finger_count += 1

                # Thumb (special case)
                if lm[4].x < lm[3].x:
                    finger_count += 1

                cv2.putText(frame, f"Fingers: {finger_count}",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 0), 2)

                # ---------------- GESTURE RECOGNITION ----------------
                thumb_tip = lm[4]
                index_tip = lm[8]
                middle_tip = lm[12]
                ring_tip = lm[16]
                pinky_tip = lm[20]

                thumb_ip = lm[3]
                index_pip = lm[6]
                middle_pip = lm[10]
                ring_pip = lm[14]
                pinky_pip = lm[18]

                # Thumbs Up
                if (thumb_tip.y < thumb_ip.y and
                    index_tip.y > index_pip.y and
                    middle_tip.y > middle_pip.y and
                    ring_tip.y > ring_pip.y and
                    pinky_tip.y > pinky_pip.y):

                    cv2.putText(frame, "Thumbs Up 👍",
                                (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

                # Peace
                elif (index_tip.y < index_pip.y and
                      middle_tip.y < middle_pip.y and
                      ring_tip.y > ring_pip.y and
                      pinky_tip.y > pinky_pip.y):

                    cv2.putText(frame, "Peace ✌",
                                (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2)

                # Rock Sign
                elif (index_tip.y < index_pip.y and
                      middle_tip.y > middle_pip.y and
                      ring_tip.y > ring_pip.y and
                      pinky_tip.y < pinky_pip.y):

                    cv2.putText(frame, "Rock Sign 🤟",
                                (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

        cv2.imshow("AI Smart Detection System", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
