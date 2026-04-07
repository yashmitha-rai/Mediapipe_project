import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import math

# ─────────────────────────────────────────────
#  PYGAME MUSIC SETUP
# ─────────────────────────────────────────────
pygame.mixer.init()

MUSIC_FILE = "calming_music.mp3"   # ← Put your MP3 file here (same folder)

def play_music():
    if os.path.exists(MUSIC_FILE):
        pygame.mixer.music.load(MUSIC_FILE)
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(-1)  # loop forever
    else:
        print(f"[INFO] Music file '{MUSIC_FILE}' not found. Add an MP3 named 'calming_music.mp3'.")

def stop_music():
    pygame.mixer.music.stop()

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose     = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────
# Eye landmarks (MediaPipe Face Mesh)
LEFT_EYE_TOP, LEFT_EYE_BOTTOM   = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
LEFT_EYE_L, LEFT_EYE_R          = 33,  133
RIGHT_EYE_L, RIGHT_EYE_R        = 362, 263

# Eyebrow landmarks
L_BROW_TOP, L_BROW_BOT = 105, 52
R_BROW_TOP, R_BROW_BOT = 334, 282

# Mouth
MOUTH_TOP, MOUTH_BOT = 13, 14

# Pose shoulders
L_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
R_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, top_idx, bot_idx, l_idx, r_idx, w, h):
    top = landmarks[top_idx]; bot = landmarks[bot_idx]
    l   = landmarks[l_idx];   r   = landmarks[r_idx]
    vert = math.dist((top.x*w, top.y*h), (bot.x*w, bot.y*h))
    horiz= math.dist((l.x*w,   l.y*h),   (r.x*w,   r.y*h))
    return vert / (horiz + 1e-6)

EAR_CLOSED_THRESH = 0.18   # below → eyes closed

def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, thickness)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, thickness)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def score_color(score):
    if score >= 75: return (0, 220, 120)
    if score >= 45: return (0, 180, 255)
    return (60, 60, 255)

# ─────────────────────────────────────────────
#  TIMER SELECTION  (simple CV window)
# ─────────────────────────────────────────────
def select_timer():
    options = [5, 10, 20]
    selected = [1]   # default 10 min

    win = "🧘 Select Meditation Duration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 520, 300)

    while True:
        canvas = np.zeros((300, 520, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 28)

        cv2.putText(canvas, "SELECT MEDITATION DURATION",
                    (60, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 140, 255), 1)

        for i, mins in enumerate(options):
            x = 60 + i * 150
            if i == selected[0]:
                draw_rounded_rect(canvas, x-5, 90, 120, 70, 12, (70, 50, 140), -1, 0.9)
                border_col = (160, 120, 255)
            else:
                border_col = (80, 80, 100)
            cv2.rectangle(canvas, (x-5, 90), (x+115, 160), border_col, 2)
            cv2.putText(canvas, f"{mins} min",
                        (x+10, 135), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                        (255,255,255) if i == selected[0] else (150,150,170), 2)

        cv2.putText(canvas, "Keys:  1 = 5min   2 = 10min   3 = 20min",
                    (55, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,140), 1)
        cv2.putText(canvas, "Press  ENTER  to start",
                    (150, 260), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 120), 2)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('1'): selected[0] = 0
        if key == ord('2'): selected[0] = 1
        if key == ord('3'): selected[0] = 2
        if key in (13, 10):   # ENTER
            break
        if key == 27:         # ESC → quit
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow(win)
    return options[selected[0]] * 60   # seconds

# ─────────────────────────────────────────────
#  MAIN SESSION
# ─────────────────────────────────────────────
def run_session(duration_sec):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    play_music()

    start_time   = time.time()
    score        = 50.0
    score_history= []

    # For stillness
    prev_pose_pts= None

    # Breathing tracker
    shoulder_y_history = []
    BREATH_WINDOW = 60   # frames

    # Score component histories (smoothing)
    eye_scores   = []
    still_scores = []
    calm_scores  = []
    breath_scores= []

    WIN = "🧘 Meditation Focus Tracker"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 560)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        elapsed   = time.time() - start_time
        remaining = max(0, duration_sec - elapsed)
        if remaining == 0:
            break

        # ── Run MediaPipe ──
        face_res = face_mesh.process(rgb)
        pose_res = pose.process(rgb)

        # ── SCORES ──
        eye_score   = 0.0
        still_score = 0.0
        calm_score  = 0.0
        breath_score= 0.0

        # 1. EYE CLOSURE
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark
            l_ear = eye_aspect_ratio(lm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,
                                     LEFT_EYE_L,  LEFT_EYE_R,  w, h)
            r_ear = eye_aspect_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                     RIGHT_EYE_L, RIGHT_EYE_R, w, h)
            avg_ear = (l_ear + r_ear) / 2
            eye_score = 100.0 if avg_ear < EAR_CLOSED_THRESH else max(0, (EAR_CLOSED_THRESH - avg_ear) / EAR_CLOSED_THRESH * 100 + 50)

            # 2. FACIAL CALMNESS (eyebrow + mouth relaxed)
            brow_dist  = abs(lm[L_BROW_TOP].y - lm[L_BROW_BOT].y) * h
            mouth_open = abs(lm[MOUTH_TOP].y  - lm[MOUTH_BOT].y)  * h
            calm_score = 100.0 if (brow_dist < 15 and mouth_open < 12) else 50.0

        # 3. BODY STILLNESS
        if pose_res.pose_landmarks:
            pts = np.array([[lm.x * w, lm.y * h]
                            for lm in pose_res.pose_landmarks.landmark])
            if prev_pose_pts is not None:
                movement = np.mean(np.linalg.norm(pts - prev_pose_pts, axis=1))
                still_score = max(0, 100 - movement * 15)
            prev_pose_pts = pts

            # 4. BREATHING RHYTHM (shoulder Y oscillation)
            l_sh = pose_res.pose_landmarks.landmark[L_SHOULDER.value]
            r_sh = pose_res.pose_landmarks.landmark[R_SHOULDER.value]
            avg_sh_y = (l_sh.y + r_sh.y) / 2
            shoulder_y_history.append(avg_sh_y)
            if len(shoulder_y_history) > BREATH_WINDOW:
                shoulder_y_history.pop(0)
            if len(shoulder_y_history) == BREATH_WINDOW:
                amp = (max(shoulder_y_history) - min(shoulder_y_history)) * h
                # calm breathing: small, regular oscillation (2-8px)
                breath_score = 100.0 if 2 < amp < 10 else max(0, 100 - abs(amp - 6) * 10)

        # ── SMOOTH SCORES ──
        def smooth(hist, val, n=15):
            hist.append(val); 
            if len(hist) > n: hist.pop(0)
            return np.mean(hist)

        es = smooth(eye_scores,   eye_score)
        ss = smooth(still_scores, still_score)
        cs = smooth(calm_scores,  calm_score)
        bs = smooth(breath_scores,breath_score)

        raw_score = 0.35*es + 0.30*ss + 0.20*cs + 0.15*bs
        score = 0.92 * score + 0.08 * raw_score   # inertia
        score_history.append(score)

        # ─────────── DRAW UI ───────────
        # Dark overlay panel on left
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (260, h), (12,12,20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # ── Score ring ──
        cx, cy, radius = 130, 130, 90
        col = score_color(score)
        cv2.circle(frame, (cx,cy), radius, (40,40,55), 14)
        angle = int(score / 100 * 360)
        for a in range(0, angle, 2):
            rad = math.radians(a - 90)
            x1 = int(cx + (radius)*math.cos(rad))
            y1 = int(cy + (radius)*math.sin(rad))
            cv2.circle(frame, (x1,y1), 7, col, -1)
        cv2.putText(frame, f"{int(score)}",
                    (cx-28, cy+12), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255,255,255), 2)
        cv2.putText(frame, "FOCUS",
                    (cx-22, cy+35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,200), 1)

        # ── Grade ──
        grade = "🥇 DEEP FOCUS" if score>=75 else "🥈 MODERATE" if score>=45 else "🥉 DISTRACTED"
        g_col = (0,220,120) if score>=75 else (0,180,255) if score>=45 else (60,60,255)
        cv2.putText(frame, grade, (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_col, 1)

        # ── Component bars ──
        bars = [("Eyes",  int(es), (100,200,255)),
                ("Still", int(ss), (100,255,160)),
                ("Calm",  int(cs), (255,200,100)),
                ("Breath",int(bs), (200,100,255))]
        for i, (lbl, val, bcol) in enumerate(bars):
            bx, by = 10, 270 + i*45
            cv2.putText(frame, lbl, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,200), 1)
            cv2.rectangle(frame, (bx, by+6), (bx+220, by+18), (40,40,55), -1)
            cv2.rectangle(frame, (bx, by+6), (bx+int(220*val/100), by+18), bcol, -1)
            cv2.putText(frame, f"{val}%", (bx+225, by+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, bcol, 1)

        # ── Timer ──
        mins = int(remaining)//60
        secs = int(remaining)%60
        timer_str = f"{mins:02d}:{secs:02d}"
        t_col = (0,220,120) if remaining > 60 else (60,60,255)
        cv2.putText(frame, timer_str, (10, h-60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, t_col, 2)
        cv2.putText(frame, "remaining", (10, h-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,140), 1)

        # ── Mini score graph ──
        gh, gw = 80, 220
        gx, gy = 10, h - 180
        cv2.rectangle(frame, (gx,gy), (gx+gw, gy+gh), (30,30,45), -1)
        if len(score_history) > 2:
            step = max(1, len(score_history)//gw)
            pts_g = [(gx + int(i*gw/len(score_history)),
                      gy + gh - int(s/100*gh))
                     for i,s in enumerate(score_history[::step])]
            for j in range(1, len(pts_g)):
                cv2.line(frame, pts_g[j-1], pts_g[j], (100,180,255), 1)

        cv2.putText(frame, "ESC to end session", (w-220, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,100), 1)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

    # ─────────── SESSION SUMMARY ───────────
    stop_music()
    cap.release()
    cv2.destroyWindow(WIN)

    avg_score = np.mean(score_history) if score_history else 0
    final_grade = "🥇 DEEP FOCUS" if avg_score>=75 else "🥈 MODERATE" if avg_score>=45 else "🥉 DISTRACTED"
    actual_mins = int(elapsed)//60
    actual_secs = int(elapsed)%60

    summary = np.zeros((400, 540, 3), dtype=np.uint8)
    summary[:] = (18,18,28)
    cv2.putText(summary, "SESSION COMPLETE", (100,60),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (160,120,255), 2)
    cv2.putText(summary, f"Duration : {actual_mins}m {actual_secs}s",
                (80,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,220), 1)
    cv2.putText(summary, f"Avg Score: {int(avg_score)} / 100",
                (80,165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,220), 1)
    cv2.putText(summary, f"Grade    : {final_grade}",
                (80,210), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0,220,120) if avg_score>=75 else (0,180,255) if avg_score>=45 else (60,60,255), 1)
    cv2.putText(summary, "Press any key to exit",
                (140,320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,100,120), 1)

    cv2.imshow("Session Summary", summary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    duration = select_timer()
    run_session(duration)