import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_det, \
     mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_result = hands.process(rgb)
        face_result = face_det.process(rgb)
        mesh_result = face_mesh.process(rgb)

        # -------- FACE DETECTION --------
        if face_result.detections:
            cv2.putText(frame, "Person Detected",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3)

            for detection in face_result.detections:
                mp_drawing.draw_detection(frame, detection)

        # -------- BLINK DETECTION --------
        if mesh_result.multi_face_landmarks:
            for face_landmarks in mesh_result.multi_face_landmarks:

                lm = face_landmarks.landmark

                # Left eye
                left_top = lm[159]
                left_bottom = lm[145]

                # Right eye
                right_top = lm[386]
                right_bottom = lm[374]

                left_dist = abs(left_top.y - left_bottom.y)
                right_dist = abs(right_top.y - right_bottom.y)

                # Blink threshold
                if left_dist < 0.01 and right_dist < 0.01:
                    cv2.putText(frame, "Blink Detected",
                                (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 3)

        # -------- HAND DETECTION --------
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[0]

                # Hand raised condition
                if wrist.y < 0.5:
                    cv2.putText(frame, "Student wants to speak",
                                (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

        cv2.imshow("Hand + Face + Blink", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):   # ESC or q
            break

cap.release()
cv2.destroyAllWindows()
