import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from collections import Counter

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_L  = 263
RIGHT_EYE_R = 33
LEFT_MOUTH  = 61
RIGHT_MOUTH = 291

total_frames      = 0
attentive_frames  = 0
distracted_frames = 0
drowsy_frames     = 0
absent_frames     = 0
blink_count       = 0
eye_closed_start  = None
eye_was_closed    = False
distraction_log   = []
session_start     = time.time()
status_buffer     = []

# ---- THRESHOLDS (relaxed for natural movement) ----
EAR_THRESHOLD   = 0.18
DROWSY_SECONDS  = 3.0
YAW_THRESHOLD   = 35
PITCH_THRESHOLD = 30
BUFFER_SIZE     = 15

def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    pts = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks, frame_w, frame_h):
    image_points = np.array([
        (landmarks[NOSE_TIP].x * frame_w,    landmarks[NOSE_TIP].y * frame_h),
        (landmarks[CHIN].x * frame_w,         landmarks[CHIN].y * frame_h),
        (landmarks[LEFT_EYE_L].x * frame_w,   landmarks[LEFT_EYE_L].y * frame_h),
        (landmarks[RIGHT_EYE_R].x * frame_w,  landmarks[RIGHT_EYE_R].y * frame_h),
        (landmarks[LEFT_MOUTH].x * frame_w,   landmarks[LEFT_MOUTH].y * frame_h),
        (landmarks[RIGHT_MOUTH].x * frame_w,  landmarks[RIGHT_MOUTH].y * frame_h),
    ], dtype="double")

    model_points = np.array([
        (0.0,    0.0,    0.0),
        (0.0,   -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0,  170.0, -135.0),
        (-150.0,-150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length  = frame_w
    center        = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    return angles[0], angles[1]

def draw_bar(frame, x, y, w, h, value, max_val, color, label):
    pct = min(value / max_val, 1.0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * pct), y + h), color, -1)
    cv2.putText(frame, f"{label}: {int(pct*100)}%", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

def draw_panel(frame, x, y, w, h, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

print("Starting... Press ESC to stop")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    total_frames += 1
    now     = time.time()
    elapsed = int(now - session_start)

    raw_status   = "ABSENT"
    status_color = (100, 100, 100)
    reason       = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # Blink / drowsy detection
            left_ear   = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear  = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear        = (left_ear + right_ear) / 2.0
            eye_closed = ear < EAR_THRESHOLD

            if eye_closed:
                if not eye_was_closed:
                    eye_closed_start = now
                    eye_was_closed   = True
                closed_duration = now - eye_closed_start
            else:
                if eye_was_closed:
                    blink_count += 1
                eye_was_closed  = False
                closed_duration = 0

            # Head pose
            try:
                pitch, yaw = get_head_pose(lm, w, h)
            except:
                pitch, yaw = 0, 0

            # Raw status decision
            if eye_closed and closed_duration >= DROWSY_SECONDS:
                raw_status = "DROWSY"
                reason     = "Eyes closed"
            elif abs(yaw) > YAW_THRESHOLD:
                raw_status = "DISTRACTED"
                reason     = f"Looking {'left' if yaw < 0 else 'right'}"
            elif pitch < -PITCH_THRESHOLD:
                raw_status = "DISTRACTED"
                reason     = "Head down"
            elif pitch > PITCH_THRESHOLD:
                raw_status = "DISTRACTED"
                reason     = "Looking up"
            else:
                raw_status = "ATTENTIVE"
                reason     = ""

            mp_drawing.draw_landmarks(
                frame, face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
    else:
        raw_status = "ABSENT"
        reason     = "No face detected"

    # ---- SMOOTHING BUFFER ----
    status_buffer.append(raw_status)
    if len(status_buffer) > BUFFER_SIZE:
        status_buffer.pop(0)
    status = Counter(status_buffer).most_common(1)[0][0]

    # Count frames based on smoothed status
    if status == "ATTENTIVE":
        attentive_frames  += 1
        status_color = (0, 220, 0)
    elif status == "DISTRACTED":
        distracted_frames += 1
        status_color = (0, 0, 255)
    elif status == "DROWSY":
        drowsy_frames     += 1
        status_color = (0, 100, 255)
    else:
        absent_frames     += 1
        status_color = (100, 100, 100)

    # Log distractions
    if status in ("DISTRACTED", "DROWSY", "ABSENT"):
        if not distraction_log or (now - distraction_log[-1][0]) > 3:
            distraction_log.append((now, reason))

    # ---- UI ----
    draw_panel(frame, 10, 10, 320, 235)

    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Status: {status}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
    if reason:
        cv2.putText(frame, reason, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if total_frames > 0:
        draw_bar(frame, 20, 110, 280, 16, attentive_frames,  total_frames, (0, 220, 0),   "Attentive")
        draw_bar(frame, 20, 147, 280, 16, distracted_frames, total_frames, (0, 0, 255),   "Distracted")
        draw_bar(frame, 20, 184, 280, 16, drowsy_frames,     total_frames, (0, 100, 255), "Drowsy")

    cv2.putText(frame, f"Blinks: {blink_count}", (20, 228),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "ESC = Stop & Report", (w - 220, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    cv2.imshow("Student Attention Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---- FINAL REPORT ----
total    = max(total_frames, 1)
att_pct  = round(attentive_frames  / total * 100, 1)
dis_pct  = round(distracted_frames / total * 100, 1)
dro_pct  = round(drowsy_frames     / total * 100, 1)
abs_pct  = round(absent_frames     / total * 100, 1)
duration = int(time.time() - session_start)
mins, secs = divmod(duration, 60)

print("\n" + "="*50)
print("       STUDENT ATTENTION REPORT")
print("="*50)
print(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Duration   : {mins} min {secs} sec")
print(f"  Blinks     : {blink_count}")
print("-"*50)
print(f"  Attentive  : {att_pct}%")
print(f"  Distracted : {dis_pct}%")
print(f"  Drowsy     : {dro_pct}%")
print(f"  Absent     : {abs_pct}%")
print("-"*50)
print(f"  Distraction Events ({len(distraction_log)}):")
for t, r in distraction_log[:10]:
    elapsed_at = int(t - session_start)
    m, s = divmod(elapsed_at, 60)
    print(f"    [{m:02d}:{s:02d}] {r}")
if len(distraction_log) > 10:
    print(f"    ... and {len(distraction_log)-10} more")
print("="*50)
if att_pct >= 75:
    print("  Result: GOOD ATTENTION")
elif att_pct >= 50:
    print("  Result: MODERATE ATTENTION")
else:
    print("  Result: POOR ATTENTION")
print("="*50)