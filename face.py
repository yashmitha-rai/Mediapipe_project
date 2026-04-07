import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                lm = face_landmarks.landmark

                # -----------------------------
                # DRAW FULL FACE MESH
                # -----------------------------
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # -----------------------------
                # IMPORTANT LANDMARKS
                # -----------------------------
                nose = lm[1]
                left_cheek = lm[234]
                right_cheek = lm[454]
                chin = lm[152]
                forehead = lm[10]

                # -----------------------------
                # HEAD LEFT / RIGHT
                # -----------------------------
                face_center_x = (left_cheek.x + right_cheek.x) / 2

                if nose.x < face_center_x - 0.04:
                    cv2.putText(frame, "Looking Left ⬅",
                                (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 3)

                elif nose.x > face_center_x + 0.04:
                    cv2.putText(frame, "Looking Right ➡",
                                (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 3)

                # -----------------------------
                # HEAD UP / DOWN
                # -----------------------------
                face_center_y = (forehead.y + chin.y) / 2

                if nose.y < face_center_y - 0.04:
                    cv2.putText(frame, "Looking Up ⬆",
                                (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 0), 3)

                elif nose.y > face_center_y + 0.04:
                    cv2.putText(frame, "Looking Down ⬇",
                                (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 0), 3)

        cv2.imshow("Head Direction Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
