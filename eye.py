import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# ===== SETTINGS =====
BLINK_THRESHOLD = 0.22
CLOSED_FRAMES_THRESHOLD = 5
DISPLAY_FRAMES = 12
DROWSY_FRAMES_THRESHOLD = 50   # ~1.5-2 seconds for alert

blink_counter = 0
closed_counter = 0
drowsy_counter = 0

display_status = "Open"
display_counter = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # REQUIRED for iris
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        current_status = "Open"
        direction = "Center"

        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [w, h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])

            # LEFT EYE
            L_LEFT = mesh_points[33]
            L_RIGHT = mesh_points[133]
            L_TOP = mesh_points[159]
            L_BOTTOM = mesh_points[145]
            L_IRIS = mesh_points[468]

            # RIGHT EYE
            R_LEFT = mesh_points[362]
            R_RIGHT = mesh_points[263]
            R_TOP = mesh_points[386]
            R_BOTTOM = mesh_points[374]
            R_IRIS = mesh_points[473]

            # ===== EAR calculation =====
            L_ear = np.linalg.norm(L_TOP - L_BOTTOM) / np.linalg.norm(L_LEFT - L_RIGHT)
            R_ear = np.linalg.norm(R_TOP - R_BOTTOM) / np.linalg.norm(R_LEFT - R_RIGHT)

            left_closed = L_ear < BLINK_THRESHOLD
            right_closed = R_ear < BLINK_THRESHOLD

            # ===== STATUS LOGIC =====
            if left_closed and right_closed:
                closed_counter += 1
                blink_counter += 1
                drowsy_counter += 1  # count drowsiness frames

                if closed_counter >= CLOSED_FRAMES_THRESHOLD:
                    current_status = "Closed"
                else:
                    current_status = "Blinked"

            else:
                if 1 < blink_counter < CLOSED_FRAMES_THRESHOLD:
                    current_status = "Blinked"
                else:
                    current_status = "Open"

                closed_counter = 0
                blink_counter = 0
                drowsy_counter = 0  # reset drowsy when eyes open

            # ===== DISPLAY HOLD =====
            if current_status != "Open":
                display_status = current_status
                display_counter = DISPLAY_FRAMES
            elif display_counter > 0:
                display_counter -= 1
            else:
                display_status = "Open"

            # ===== DRAW IRIS =====
            cv2.circle(frame, tuple(L_IRIS), 3, (0, 255, 0), -1)
            cv2.circle(frame, tuple(R_IRIS), 3, (0, 255, 0), -1)

            # Draw eye lines
            cv2.line(frame, tuple(L_LEFT), tuple(L_RIGHT), (255, 0, 0), 1)
            cv2.line(frame, tuple(R_LEFT), tuple(R_RIGHT), (255, 0, 0), 1)

            # ===== IRIS DIRECTION (ONLY WHEN OPEN) =====
            if display_status == "Open":
                left_width = L_RIGHT[0] - L_LEFT[0]
                right_width = R_RIGHT[0] - R_LEFT[0]

                left_ratio = (L_IRIS[0] - L_LEFT[0]) / left_width
                right_ratio = (R_IRIS[0] - R_LEFT[0]) / right_width

                avg_ratio = (left_ratio + right_ratio) / 2

                if avg_ratio < 0.35:
                    direction = "Left"
                elif avg_ratio > 0.65:
                    direction = "Right"
                else:
                    direction = "Center"

            # ===== DROWSINESS ALERT =====
            if drowsy_counter >= DROWSY_FRAMES_THRESHOLD:
                cv2.putText(frame, "ALERT!", (200, 200),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

        # ===== DISPLAY TEXT =====
        status_color = (0, 0, 255) if display_status != "Open" else (0, 255, 0)

        cv2.putText(frame, f"STATUS: {display_status}",
                    (30, 80),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2, status_color, 2)

        cv2.putText(frame, f"DIRECTION: {direction}",
                    (30, 130),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2, (255, 0, 0), 2)

        cv2.imshow("Eye Status + Iris + Direction + Drowsiness Alert", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()