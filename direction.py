import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

distraction_count = 0
score = 100
last_state = "Focused"
cooldown_time = 1.0
last_distraction_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    state = "Focused"

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        # ---- IMPORTANT LANDMARKS ----
        nose = face.landmark[1]
        left_eye_corner = face.landmark[33]
        right_eye_corner = face.landmark[263]
        left_eye_top = face.landmark[159]
        left_eye_bottom = face.landmark[145]
        mouth_top = face.landmark[13]
        mouth_bottom = face.landmark[14]

        # ------------------------------
        # 1️⃣ LEFT / RIGHT DETECTION
        face_center = (left_eye_corner.x + right_eye_corner.x) / 2
        diff = nose.x - face_center
        lr_threshold = 0.035

        if diff > lr_threshold:
            state = "Looking Right"
        elif diff < -lr_threshold:
            state = "Looking Left"

        # ------------------------------
        # 2️⃣ LOOKING DOWN DETECTION
        down_threshold = 0.06
        if nose.y - left_eye_corner.y > down_threshold:
            state = "Looking Down"

        # ------------------------------
        # 3️⃣ BLINK DETECTION
        eye_height = abs(left_eye_top.y - left_eye_bottom.y)
        blink_threshold = 0.012

        if eye_height < blink_threshold:
            state = "Blinking"

        # ------------------------------
        # 4️⃣ YAWN DETECTION
        mouth_open = abs(mouth_top.y - mouth_bottom.y)
        yawn_threshold = 0.065

        if mouth_open > yawn_threshold:
            state = "Yawning"

        # ------------------------------
        # DISTRACTION COUNT LOGIC
        current_time = time.time()

        if state != "Focused" and last_state == "Focused":
            if current_time - last_distraction_time > cooldown_time:
                distraction_count += 1
                score -= 5
                last_distraction_time = current_time

        last_state = state

    # Keep score between 0 and 100
    score = max(0, min(100, score))

    # ------------------------------
    # DISPLAY
    cv2.putText(frame, f"State: {state}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(frame, f"Distractions: {distraction_count}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.putText(frame, f"Attention Score: {score}%", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("AI Attention Drift Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()