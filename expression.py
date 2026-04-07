import cv2
import mediapipe as mp
import numpy as np
from collections import deque

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

# ---- LANDMARK INDICES ----
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33, 160, 158, 133, 153, 144]

LEFT_EYEBROW  = [336, 296, 334, 293, 300]
RIGHT_EYEBROW = [107, 66, 105, 63, 70]

LEFT_MOUTH_CORNER  = 61
RIGHT_MOUTH_CORNER = 291
UPPER_LIP = 13
LOWER_LIP = 14

# Head nod tracking
prev_pitch    = None
nod_count     = 0
nod_direction = None
nod_cooldown  = 0
nod_detected  = ""

# Smoothing buffers
smile_buffer = deque(maxlen=10)
brow_buffer  = deque(maxlen=10)

# ---- HELPER FUNCTIONS ----
def get_pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h])

def dist(a, b):
    return np.linalg.norm(a - b)

def ear(lm, indices, w, h):
    pts = [get_pt(lm, i, w, h) for i in indices]
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def get_head_pose(lm, w, h):
    image_points = np.array([
        (lm[1].x * w,   lm[1].y * h),
        (lm[152].x * w, lm[152].y * h),
        (lm[263].x * w, lm[263].y * h),
        (lm[33].x * w,  lm[33].y * h),
        (lm[61].x * w,  lm[61].y * h),
        (lm[291].x * w, lm[291].y * h),
    ], dtype="double")

    model_points = np.array([
        (0.0,    0.0,    0.0),
        (0.0,   -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0,  170.0, -135.0),
        (-150.0,-150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length  = w
    camera_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ], dtype="double")

    _, rvec, _ = cv2.solvePnP(model_points, image_points,
                               camera_matrix, np.zeros((4,1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _  = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1]

def draw_panel(frame, x, y, w, h, alpha=0.7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def draw_status(frame, x, y, label, value, active_color):
    color = active_color if value else (80, 80, 80)
    icon  = "●" if value else "○"
    cv2.putText(frame, f"{icon} {label}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

print("Face Expression Detector Started! Press ESC to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    is_smiling     = False
    is_left_wink   = False
    is_right_wink  = False
    is_brow_raised = False
    smile_ratio    = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # ---- 1. SMILE DETECTION (Improved) ----
            left_corner  = get_pt(lm, LEFT_MOUTH_CORNER,  w, h)
            right_corner = get_pt(lm, RIGHT_MOUTH_CORNER, w, h)
            upper_lip    = get_pt(lm, UPPER_LIP, w, h)
            lower_lip    = get_pt(lm, LOWER_LIP, w, h)

            mouth_width  = dist(left_corner, right_corner)
            mouth_height = dist(upper_lip, lower_lip)

            face_width = dist(get_pt(lm, 234, w, h),
                              get_pt(lm, 454, w, h))

            width_ratio  = mouth_width / face_width
            height_ratio = mouth_height / face_width

            smile_raw = width_ratio > 0.28 and height_ratio > 0.02

            smile_buffer.append(smile_raw)
            is_smiling = sum(smile_buffer) > len(smile_buffer) * 0.6

            smile_ratio = width_ratio

            # ---- 2. WINK DETECTION ----
            left_ear_val  = ear(lm, LEFT_EYE,  w, h)
            right_ear_val = ear(lm, RIGHT_EYE, w, h)

            is_left_wink  = left_ear_val  < 0.13 and right_ear_val > 0.20
            is_right_wink = right_ear_val < 0.13 and left_ear_val  > 0.20

            # ---- 3. EYEBROW RAISE ----
            face_height = dist(get_pt(lm, 10, w, h),
                               get_pt(lm, 152, w, h))

            left_eye_center   = np.mean([get_pt(lm, i, w, h) for i in LEFT_EYE],  axis=0)
            right_eye_center  = np.mean([get_pt(lm, i, w, h) for i in RIGHT_EYE], axis=0)
            left_brow_center  = np.mean([get_pt(lm, i, w, h) for i in LEFT_EYEBROW],  axis=0)
            right_brow_center = np.mean([get_pt(lm, i, w, h) for i in RIGHT_EYEBROW], axis=0)

            left_brow_norm  = dist(left_eye_center,  left_brow_center)  / face_height
            right_brow_norm = dist(right_eye_center, right_brow_center) / face_height
            avg_brow_norm   = (left_brow_norm + right_brow_norm) / 2

            brow_buffer.append(avg_brow_norm > 0.18)
            is_brow_raised = sum(brow_buffer) > len(brow_buffer) * 0.6

            # ---- 4. HEAD NOD ----
            try:
                pitch, yaw = get_head_pose(lm, w, h)
            except:
                pitch, yaw = 0, 0

            if prev_pitch is not None and nod_cooldown <= 0:
                pitch_diff = pitch - prev_pitch

                if pitch_diff > 3:
                    if nod_direction != "down":
                        nod_direction = "down"
                elif pitch_diff < -3:
                    if nod_direction == "down":
                        nod_count += 1
                        nod_detected = "YES NOD!"
                        nod_cooldown = 30
                        nod_direction = "up"

            prev_pitch = pitch

            if nod_cooldown > 0:
                nod_cooldown -= 1
            else:
                nod_detected = ""

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    # ---- UI PANEL ----
    draw_panel(frame, 5, 5, 300, 250)

    draw_status(frame, 15, 50,  "SMILING",        is_smiling,    (0, 255, 100))
    draw_status(frame, 15, 85,  "LEFT WINK",      is_left_wink,  (0, 200, 255))
    draw_status(frame, 15, 120, "RIGHT WINK",     is_right_wink, (0, 200, 255))
    draw_status(frame, 15, 155, "EYEBROW RAISED", is_brow_raised,(255, 200, 0))

    cv2.putText(frame, f"Nods: {nod_count}", (15, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)

    # Smile strength bar
    draw_panel(frame, 5, h-60, 300, 50)
    bar_w = int(min(smile_ratio / 0.5, 1.0) * 270)
    cv2.rectangle(frame, (15, h-48), (285, h-30), (40, 40, 40), -1)
    cv2.rectangle(frame, (15, h-48), (15+bar_w, h-30), (0, 255, 100), -1)
    cv2.putText(frame, f"Smile Strength: {int(smile_ratio*200)}%", (15, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Face Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()