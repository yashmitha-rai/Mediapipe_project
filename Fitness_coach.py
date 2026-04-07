"""
Smart Fitness Coach - CHAIR / DESK MODE
GREEN / RED rules:
  - POSTURE (all exercises): sit straight = GREEN, bend forward/back too much = RED
  - BICEP CURL: arm curled UP = GREEN, arm down = YELLOW (waiting)
  - NECK TILT:  head tilted to side = GREEN, head straight/center = YELLOW (waiting)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
CYAN   = (255, 255, 0)
ORANGE = (0, 165, 255)

# ── Landmarks ────────────────────────────────────────────────────────────────
LS   = mp_pose.PoseLandmark.LEFT_SHOULDER
RS   = mp_pose.PoseLandmark.RIGHT_SHOULDER
LE   = mp_pose.PoseLandmark.LEFT_ELBOW
RE   = mp_pose.PoseLandmark.RIGHT_ELBOW
LW   = mp_pose.PoseLandmark.LEFT_WRIST
RW   = mp_pose.PoseLandmark.RIGHT_WRIST
LH   = mp_pose.PoseLandmark.LEFT_HIP
RH   = mp_pose.PoseLandmark.RIGHT_HIP
LK   = mp_pose.PoseLandmark.LEFT_KNEE
RK   = mp_pose.PoseLandmark.RIGHT_KNEE
LA   = mp_pose.PoseLandmark.LEFT_ANKLE
RA   = mp_pose.PoseLandmark.RIGHT_ANKLE
NOSE = mp_pose.PoseLandmark.NOSE
LEAR = mp_pose.PoseLandmark.LEFT_EAR
REAR = mp_pose.PoseLandmark.RIGHT_EAR

# ── Math ─────────────────────────────────────────────────────────────────────
def angle3(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b;  bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))

def lm(lms, idx):
    return [lms[idx].x, lms[idx].y]

def lm_px(lms, idx, W, H):
    return (int(lms[idx].x * W), int(lms[idx].y * H))

# ── Drawing ───────────────────────────────────────────────────────────────────
def panel(frame, x, y, w, h, alpha=0.70):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), (15, 15, 15), -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (90, 90, 90), 1)

def t(frame, text, pos, scale=0.6, color=WHITE, bold=False):
    th = 3 if bold else 2
    cv2.putText(frame, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, th+1)
    cv2.putText(frame, text, pos,                  cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)

def arc(frame, vx, p1, p2, angle_val, color, r=40):
    v = np.array(vx)
    a1 = math.degrees(math.atan2(p1[1]-v[1], p1[0]-v[0]))
    a2 = math.degrees(math.atan2(p2[1]-v[1], p2[0]-v[0]))
    s, e = sorted([a1, a2])
    if e - s > 180: s, e = e, s + 360
    cv2.ellipse(frame, tuple(v.astype(int)), (r, r), 0, int(s), int(e), color, 2)
    mid = math.radians((s + e) / 2)
    t(frame, f"{int(angle_val)}", (int(v[0]+(r+16)*math.cos(mid)), int(v[1]+(r+16)*math.sin(mid))), 0.48, color)

def pbar(frame, x, y, w, h, val, maxv, color):
    pct = min(val / maxv, 1.0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*pct), y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), WHITE, 1)

# ─────────────────────────────────────────────────────────────────────────────
#  POSTURE CHECK
# ─────────────────────────────────────────────────────────────────────────────
POSTURE_MIN = 140
POSTURE_MAX = 210

def check_posture(lms):
    try:
        a = angle3(lm(lms, LEAR), lm(lms, LS), lm(lms, LH))
        if a < POSTURE_MIN:
            return False, a, "Sit up! Leaning too far forward"
        if a > POSTURE_MAX:
            return False, a, "Don't lean back!"
        return True, a, "Posture good"
    except Exception:
        return True, 0, ""

# ─────────────────────────────────────────────────────────────────────────────
#  EXERCISE FORM CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_bicep(lms):
    try:
        al = angle3(lm(lms, LS), lm(lms, LE), lm(lms, LW))
        ar = angle3(lm(lms, RS), lm(lms, RE), lm(lms, RW))
        avg = (al + ar) / 2
        if avg < 70:
            return True, "Great curl! Squeeze!"
        else:
            return None, "Curl your arm up for GREEN"
    except Exception:
        return None, ""

def check_shoulder_press(lms):
    try:
        al = angle3(lm(lms, LE), lm(lms, LS), lm(lms, LH))
        ar = angle3(lm(lms, RE), lm(lms, RS), lm(lms, RH))
        avg = (al + ar) / 2
        if avg > 150:
            return True, "Arms up! Full extension!"
        else:
            return None, "Press arms overhead for GREEN"
    except Exception:
        return None, ""

def check_lateral_raise(lms):
    try:
        al = angle3(lm(lms, LH), lm(lms, LS), lm(lms, LE))
        ar = angle3(lm(lms, RH), lm(lms, RS), lm(lms, RE))
        avg = (al + ar) / 2
        if avg > 75:
            return True, "Good raise! Hold it!"
        else:
            return None, "Raise arms sideways for GREEN"
    except Exception:
        return None, ""

def check_neck_tilt(lms):
    try:
        lear_y = lms[LEAR].y
        rear_y = lms[REAR].y
        diff   = abs(lear_y - rear_y)
        nose_x = lms[NOSE].x
        mid_x  = (lms[LS].x + lms[RS].x) / 2
        offset = abs(nose_x - mid_x)
        if diff > 0.04 or offset > 0.06:
            return True, "Good tilt! Hold it!"
        else:
            return None, "Tilt head sideways for GREEN"
    except Exception:
        return None, ""

def check_squat(lms):
    try:
        al = angle3(lm(lms, LH), lm(lms, LK), lm(lms, LA))
        ar = angle3(lm(lms, RH), lm(lms, RK), lm(lms, RA))
        avg = (al + ar) / 2
        if avg < 110:
            return True, "Good depth! Hold it!"
        else:
            return None, "Bend knees lower for GREEN"
    except Exception:
        return None, ""

def check_pushup(lms):
    try:
        al = angle3(lm(lms, LS), lm(lms, LE), lm(lms, LW))
        ar = angle3(lm(lms, RS), lm(lms, RE), lm(lms, RW))
        avg = (al + ar) / 2
        if avg < 90:
            return True, "Chest down! Great!"
        else:
            return None, "Lower your chest for GREEN"
    except Exception:
        return None, ""

CHAIR_EXERCISES = {
    "Bicep Curl": {
        "emoji": "Bicep", "desc": "Curl up = GREEN  |  Arm down = YELLOW",
        "joints_L": [LS, LE, LW], "joints_R": [RS, RE, RW],
        "down_angle": 155, "up_angle": 40,
        "check_fn": check_bicep,
        "cues": ["Extend fully at bottom", "Squeeze hard at top", "Slow on way down"],
    },
    "Shoulder Press": {
        "emoji": "Press", "desc": "Press overhead = GREEN  |  Arms down = YELLOW",
        "joints_L": [LE, LS, LH], "joints_R": [RE, RS, RH],
        "down_angle": 70, "up_angle": 155,
        "check_fn": check_shoulder_press,
        "cues": ["Start at ear level", "Full lockout at top", "Control descent"],
    },
    "Lateral Raise": {
        "emoji": "Raise", "desc": "Arms at shoulder height = GREEN",
        "joints_L": [LH, LS, LE], "joints_R": [RH, RS, RE],
        "down_angle": 170, "up_angle": 80,
        "check_fn": check_lateral_raise,
        "cues": ["Lead with elbows", "Stop at shoulder height", "Slow descent"],
    },
    "Neck Tilt": {
        "emoji": "Neck1", "desc": "Head tilted sideways = GREEN  |  Center = YELLOW",
        "joints_L": [LS, NOSE, RS], "joints_R": [LS, NOSE, RS],
        "down_angle": 85, "up_angle": 55,
        "check_fn": check_neck_tilt,
        "cues": ["Ear toward shoulder", "Keep shoulders still", "Hold 2 seconds"],
    },
    "Neck Tilt 2": {
        "emoji": "Neck2", "desc": "Head tilted sideways = GREEN  |  Center = YELLOW",
        "joints_L": [LS, NOSE, RS], "joints_R": [LS, NOSE, RS],
        "down_angle": 85, "up_angle": 55,
        "check_fn": check_neck_tilt,
        "cues": ["Ear toward shoulder", "Keep shoulders still", "Hold 2 seconds"],
    },
}

FULLBODY_EXERCISES = {
    "Bicep Curl": {
        "emoji": "Bicep", "desc": "Curl up = GREEN  |  Arm down = YELLOW",
        "joints_L": [LS, LE, LW], "joints_R": [RS, RE, RW],
        "down_angle": 155, "up_angle": 40,
        "check_fn": check_bicep,
        "cues": ["Extend fully at bottom", "Squeeze hard at top", "Slow on way down"],
    },
    "Shoulder Press": {
        "emoji": "Press", "desc": "Press overhead = GREEN  |  Arms down = YELLOW",
        "joints_L": [LE, LS, LH], "joints_R": [RE, RS, RH],
        "down_angle": 70, "up_angle": 155,
        "check_fn": check_shoulder_press,
        "cues": ["Start at ear level", "Full lockout at top", "Control descent"],
    },
    "Lateral Raise": {
        "emoji": "Raise", "desc": "Arms at shoulder height = GREEN",
        "joints_L": [LH, LS, LE], "joints_R": [RH, RS, RE],
        "down_angle": 170, "up_angle": 80,
        "check_fn": check_lateral_raise,
        "cues": ["Lead with elbows", "Stop at shoulder height", "Slow descent"],
    },
    "Squat": {
        "emoji": "Squat", "desc": "Bend knees deep = GREEN  |  Standing = YELLOW",
        "joints_L": [LH, LK, LA], "joints_R": [RH, RK, RA],
        "down_angle": 170, "up_angle": 90,
        "check_fn": check_squat,
        "cues": ["Chest up", "Knees over toes", "Go below parallel"],
    },
    "Push-Up": {
        "emoji": "Push", "desc": "Chest down = GREEN  |  Arms straight = YELLOW",
        "joints_L": [LS, LE, LW], "joints_R": [RS, RE, RW],
        "down_angle": 160, "up_angle": 70,
        "check_fn": check_pushup,
        "cues": ["Core tight", "Chest to floor", "Full lockout at top"],
    },
    "Neck Tilt": {
        "emoji": "Neck", "desc": "Head tilted sideways = GREEN  |  Center = YELLOW",
        "joints_L": [LS, NOSE, RS], "joints_R": [LS, NOSE, RS],
        "down_angle": 85, "up_angle": 55,
        "check_fn": check_neck_tilt,
        "cues": ["Ear toward shoulder", "Keep shoulders still", "Hold 2 seconds"],
    },
}

# Active exercise dict
EXERCISES = CHAIR_EXERCISES

# ── Rep Counter ───────────────────────────────────────────────────────────────
class RepCounter:
    def __init__(self):
        self.count   = 0
        self.stage   = "down"
        self.t_start = time.time()
        self.rep_ts  = []

    def update(self, angle, ex_name):
        ex = EXERCISES[ex_name]
        prev = self.stage
        if angle > ex["down_angle"] - 15:
            self.stage = "down"
        if angle < ex["up_angle"] + 15 and prev == "down":
            self.stage = "up"
            self.count += 1
            self.rep_ts.append(time.time())

    def rpm(self):
        now = time.time()
        return len([ts for ts in self.rep_ts if now - ts < 60])

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    global EXERCISES
    ex_list    = list(EXERCISES.keys())
    ex_idx     = 0
    counter    = RepCounter()
    target     = 10
    t_splash   = time.time()
    chair_mode = True

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            H, W  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            ex_name = ex_list[ex_idx]
            ex      = EXERCISES[ex_name]

            skel_col   = YELLOW
            status_txt = "Waiting..."
            status_col = YELLOW
            form_msgs  = []
            angle_L = angle_R = None
            pts_L = pts_R = []
            posture_angle = 0

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                # 1. Posture check
                posture_ok, posture_angle, posture_msg = check_posture(lms)

                # 2. Joint angles
                try:
                    angle_L = angle3(*[lm(lms, j) for j in ex["joints_L"]])
                    pts_L   = [lm_px(lms, j, W, H) for j in ex["joints_L"]]
                except Exception: pass
                try:
                    angle_R = angle3(*[lm(lms, j) for j in ex["joints_R"]])
                    pts_R   = [lm_px(lms, j, W, H) for j in ex["joints_R"]]
                except Exception: pass

                if angle_L is not None and angle_R is not None:
                    counter.update((angle_L + angle_R) / 2, ex_name)

                # 3. Exercise check
                ex_ok, ex_msg = ex["check_fn"](lms)

                # 4. Final color decision
                if not posture_ok:
                    skel_col   = RED
                    status_txt = "FIX POSTURE!"
                    status_col = RED
                    form_msgs  = [posture_msg]
                elif ex_ok is True:
                    skel_col   = GREEN
                    status_txt = "PERFECT!"
                    status_col = GREEN
                    form_msgs  = [ex_msg]
                elif ex_ok is False:
                    skel_col   = RED
                    status_txt = "FIX FORM"
                    status_col = RED
                    form_msgs  = [ex_msg]
                else:
                    skel_col   = YELLOW
                    status_txt = "READY..."
                    status_col = YELLOW
                    form_msgs  = [ex_msg] if ex_msg else []

                # 5. Draw skeleton
                cs = mp_drawing.DrawingSpec(color=skel_col, thickness=3, circle_radius=5)
                ls = mp_drawing.DrawingSpec(color=skel_col, thickness=2, circle_radius=4)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS, ls, cs)

                if angle_L and len(pts_L) == 3:
                    arc(frame, pts_L[1], pts_L[0], pts_L[2], angle_L, skel_col)
                if angle_R and len(pts_R) == 3:
                    arc(frame, pts_R[1], pts_R[0], pts_R[2], angle_R, skel_col)

                # Posture angle label
                try:
                    sp    = lm_px(lms, LS, W, H)
                    p_col = GREEN if posture_ok else RED
                    t(frame, f"P:{int(posture_angle)}", (sp[0]-10, sp[1]-18), 0.4, p_col)
                except Exception: pass

            # ── REP COUNTER COLOR ─────────────────────────────────────────
            rc = ORANGE if counter.count >= target else GREEN
            el = int(time.time() - counter.t_start)

            # ── TOP LEFT: Mode + Rep Counter ──────────────────────────────
            mode_txt = "CHAIR MODE" if chair_mode else "FULL BODY"
            mode_col = CYAN if chair_mode else ORANGE
            t(frame, mode_txt,         (30, 38),  0.55, mode_col, bold=True)
            t(frame, "[F] toggle mode",(30, 58),  0.38, (150,150,150))

            # ── REP COUNT displayed large ─────────────────────────────────
            t(frame, str(counter.count), (30, 115), 1.6, rc, bold=True)
            t(frame, f"/ {target}",      (30, 150), 0.7, (180,180,180))

            # ── TOP CENTER: Exercise name ─────────────────────────────────
            t(frame, f"{ex['emoji']}  {ex_name}", (W//2-130, 45), 0.9, YELLOW, bold=True)

            # ── TOP RIGHT: Timer ──────────────────────────────────────────
            t(frame, f"{el//60:02d}:{el%60:02d}", (W-115, 45), 0.8, WHITE)

            # ── BOTTOM CENTER: Status ─────────────────────────────────────
            t(frame, status_txt, (W//2-150, H-55), 1.1, status_col, bold=True)

            # Form message just above status
            if form_msgs:
                t(frame, form_msgs[0], (W//2-170, H-22), 0.6, status_col)

            # ── BOTTOM: Progress bar ──────────────────────────────────────
            pct = min(counter.count / target, 1.0)
            cv2.rectangle(frame, (0, H-6), (W, H),          (50,50,50), -1)
            cv2.rectangle(frame, (0, H-6), (int(W*pct), H), rc,         -1)

            # ── CONTROLS HINT bottom-left ─────────────────────────────────
            t(frame, "N=Next  P=Prev  R=Reset  +/-=Target  Q=Quit",
              (30, H-75), 0.38, (120,120,120))

            if not results.pose_landmarks:
                t(frame, "Move closer!", (W//2-120, H//2), 1.0, YELLOW, bold=True)

            # ── SPLASH SCREEN ─────────────────────────────────────────────
            if time.time() - t_splash < 6:
                panel(frame, W//2-240, H//2-130, 480, 260)
                t(frame, "Smart Fitness Coach",                   (W//2-185, H//2-90),  0.85, CYAN,   bold=True)
                t(frame, "CHAIR / DESK MODE",                     (W//2-110, H//2-60),  0.65, YELLOW)
                t(frame, "Sit at your desk, show upper body",     (W//2-185, H//2-28),  0.54, WHITE)
                t(frame, "GREEN  = posture good + correct move",  (W//2-215, H//2+4),   0.5,  GREEN)
                t(frame, "YELLOW = posture good, waiting",        (W//2-185, H//2+30),  0.5,  YELLOW)
                t(frame, "RED    = bad posture OR bad form",      (W//2-185, H//2+56),  0.5,  RED)
                t(frame, "Bicep Curl  : curl UP for GREEN",       (W//2-200, H//2+88),  0.47, WHITE)
                t(frame, "Neck Tilt   : tilt sideways for GREEN", (W//2-200, H//2+112), 0.47, WHITE)
                t(frame, "Hunch fwd/back too much = RED always",  (W//2-200, H//2+136), 0.47, RED)

            cv2.imshow("Smart Fitness Coach", frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('n'):
                ex_idx = (ex_idx+1) % len(ex_list); counter = RepCounter()
            elif key == ord('p'):
                ex_idx = (ex_idx-1) % len(ex_list); counter = RepCounter()
            elif key == ord('f'):
                chair_mode = not chair_mode
                EXERCISES  = CHAIR_EXERCISES if chair_mode else FULLBODY_EXERCISES
                ex_list    = list(EXERCISES.keys())
                ex_idx     = 0
                counter    = RepCounter()
            elif key == ord('r'):             counter = RepCounter()
            elif key in (ord('+'), ord('=')): target = min(target+1, 50)
            elif key == ord('-'):             target = max(target-1, 1)

    cap.release()
    cv2.destroyAllWindows()
    el = int(time.time() - counter.t_start)
    print(f"\n{'='*42}")
    print(f"  Exercise : {ex_list[ex_idx]}")
    print(f"  Reps     : {counter.count}/{target}")
    print(f"  Time     : {el//60:02d}:{el%60:02d}")
    print(f"{'='*42}")

if __name__ == "__main__":
    main()