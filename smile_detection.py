import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.7,
                           min_tracking_confidence=0.7) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Mouth corner landmarks
                left = face_landmarks.landmark[61]
                right = face_landmarks.landmark[291]

                # Convert to pixel coordinates
                x1, y1 = int(left.x * w), int(left.y * h)
                x2, y2 = int(right.x * w), int(right.y * h)

                # Calculate mouth width
                distance = math.hypot(x2 - x1, y2 - y1)

                # Draw line across mouth
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Smile condition
                if distance > 60:  # Adjust if needed
                    cv2.putText(frame, "Smiled",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3)

        cv2.imshow("Smile Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
