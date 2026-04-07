"""
MoodTime — AI Mood & Screen Time Tracker
Detects mood (happy, sad, angry, neutral, surprised) in real time
and tracks how long you spend in each mood during screen time.
Uses: OpenCV + DeepFace + MediaPipe Face Mesh
SAD     → detected when head looks DOWN (nose drops below chin level)
ANGRY   → detected when eyebrows are RAISED high
"""

import cv2
import time
import datetime
import numpy as np
from collections import defaultdict
import mediapipe as mp

# Try DeepFace for mood detection
try:
    from deepface import DeepFace
    USE_DEEPFACE = True
    print("DeepFace loaded successfully!")
except ImportError:
    USE_DEEPFACE = False
    print("DeepFace not found. Running in DEMO mode...")

# ── MediaPipe Face Mesh ───────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Key landmark indices
NOSE_TIP      = 4
CHIN          = 152
LEFT_EYE_TOP  = 159
LEFT_BROW_TOP = 70    # left eyebrow top
LEFT_EYE_MID  = 33    # left eye inner corner (reference)
RIGHT_BROW_TOP= 300
RIGHT_EYE_MID = 263

def get_mesh_cues(frame):
    """
    Returns (is_looking_down, is_eyebrow_raised)
    is_looking_down   → SAD override
    is_eyebrow_raised → ANGRY override
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return False, False

    lms = result.multi_face_landmarks[0].landmark
    H, W = frame.shape[:2]

    # ── HEAD TILT DOWN (SAD) ──────────────────────────────────────────────
    # Compare nose Y vs chin Y — when looking down, nose Y approaches chin Y
    nose_y  = lms[NOSE_TIP].y
    chin_y  = lms[CHIN].y
    # Normally nose_y is well above chin_y (smaller value = higher on screen)
    # When looking down, the gap shrinks
    gap = chin_y - nose_y   # normalised 0-1
    is_looking_down = gap < 0.10   # threshold: gap shrinks when head tilts down

    # ── EYEBROW RAISE (ANGRY) ─────────────────────────────────────────────
    # Compare eyebrow top Y vs eye top Y
    # When eyebrows raised, brow moves UP (smaller Y) away from eye
    left_brow_y  = lms[LEFT_BROW_TOP].y
    left_eye_y   = lms[LEFT_EYE_TOP].y
    right_brow_y = lms[RIGHT_BROW_TOP].y
    right_eye_y  = lms[338].y   # right eye top

    brow_eye_gap = ((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) / 2
    is_eyebrow_raised = brow_eye_gap > 0.025   # raised when gap increases

    return is_looking_down, is_eyebrow_raised

# ── Colors ────────────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GRAY   = (180, 180, 180)
DARK   = (30,  30,  30)

MOOD_COLORS = {
    "happy":     (0,   220, 100),   # green
    "sad":       (200, 100, 50),    # blue-ish
    "angry":     (0,   60,  220),   # red
    "neutral":   (200, 200, 200),   # gray
    "surprise":  (0,   200, 255),   # yellow
}

MOOD_EMOJI = {
    "happy":    "HAPPY    :)",
    "sad":      "SAD      :(",
    "angry":    "ANGRY    >:(",
    "neutral":  "NEUTRAL  :|",
    "surprise": "SURPRISED :O",
}

# ── Drawing helpers ───────────────────────────────────────────────────────────
def txt(frame, text, pos, scale=0.6, color=WHITE, bold=False):
    th = 3 if bold else 2
    cv2.putText(frame, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, th+1)
    cv2.putText(frame, text, pos,                  cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)

def draw_bar(frame, x, y, w, h, pct, color, label, time_str):
    # Background
    cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 60, 60), -1)
    # Fill
    fill = int(w * min(pct, 1.0))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x+fill, y+h), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
    # Label left
    txt(frame, label, (x - 130, y + h - 4), 0.45, color)
    # Time right
    txt(frame, time_str, (x + w + 8, y + h - 4), 0.42, GRAY)

def panel(frame, x, y, w, h, alpha=0.75):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), (20, 20, 20), -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (70, 70, 70), 1)

def fmt_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"

# ── Fake mood detector (demo mode) ───────────────────────────────────────────
import random
_fake_mood = "neutral"
_fake_timer = 0

def get_fake_mood():
    global _fake_mood, _fake_timer
    _fake_timer += 1
    if _fake_timer > 60:
        _fake_mood = random.choice(["happy", "neutral", "sad", "angry", "surprise"])
        _fake_timer = 0
    return _fake_mood, random.uniform(60, 95)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Tracking
    mood_times   = defaultdict(float)   # mood → total seconds
    session_start = time.time()
    current_mood  = "neutral"
    mood_conf     = 0.0
    last_detect   = time.time()
    detect_every  = 1.5   # run detection every 1.5 seconds (performance)
    face_detected = False
    mood_history  = []    # last 5 moods for smoothing

    print("\n=== MoodTime Started ===")
    print("Press Q to quit and see your session report\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        now   = time.time()
        elapsed = now - session_start

        # ── Mood detection (every N seconds) ──────────────────────────────
        if now - last_detect > detect_every:
            last_detect = now

            # MediaPipe cues — run every frame for responsiveness
            is_looking_down, is_eyebrow_raised = get_mesh_cues(frame)

            if is_looking_down:
                # Looking down → SAD override
                current_mood  = "sad"
                mood_conf     = 90.0
                face_detected = True
            elif is_eyebrow_raised:
                # Eyebrow raised → ANGRY override
                current_mood  = "angry"
                mood_conf     = 90.0
                face_detected = True
            elif USE_DEEPFACE:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=True,
                        silent=True
                    )
                    if isinstance(result, list): result = result[0]
                    emotions     = result["emotion"]
                    current_mood = max(emotions, key=emotions.get).lower()
                    mood_conf    = emotions[current_mood]
                    face_detected = True

                    # smooth over last 5 detections
                    mood_history.append(current_mood)
                    if len(mood_history) > 5:
                        mood_history.pop(0)
                    current_mood = max(set(mood_history), key=mood_history.count)

                except Exception:
                    face_detected = False
            else:
                current_mood, mood_conf = get_fake_mood()
                face_detected = True

        # Track time in current mood
        if face_detected:
            mood_times[current_mood] += detect_every

        mood_color = MOOD_COLORS.get(current_mood, WHITE)

        # ── Draw face detection box ────────────────────────────────────────
        if face_detected:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), mood_color, 2)
                # Mood label above face box
                txt(frame, MOOD_EMOJI.get(current_mood, current_mood.upper()),
                    (x, y-12), 0.65, mood_color, bold=True)

        # ── TOP: session info bar ──────────────────────────────────────────
        panel(frame, 0, 0, W, 55)
        txt(frame, "MoodTime", (20, 36), 0.9, (0, 200, 255), bold=True)
        txt(frame, "AI Mood & Screen Time Tracker", (160, 36), 0.55, GRAY)
        txt(frame, f"Session: {fmt_time(elapsed)}", (W-220, 36), 0.6, WHITE)
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        txt(frame, now_str, (W-340, 36), 0.55, GRAY)

        # ── RIGHT: mood bars ───────────────────────────────────────────────
        panel(frame, W-280, 65, 270, 380)
        txt(frame, "MOOD BREAKDOWN", (W-270, 90), 0.5, (0, 200, 255), bold=True)

        total_tracked = max(sum(mood_times.values()), 1)
        bar_x = W - 170
        bar_w = 130
        bar_h = 18

        for i, (mood, color) in enumerate(MOOD_COLORS.items()):
            t_mood = mood_times.get(mood, 0)
            pct    = t_mood / total_tracked
            by     = 110 + i * 44
            draw_bar(frame, bar_x, by, bar_w, bar_h, pct,
                     color, mood.upper(), fmt_time(t_mood))
            # percentage
            txt(frame, f"{int(pct*100)}%", (bar_x + bar_w + 50, by + bar_h - 4), 0.4, GRAY)

        # ── LEFT: current mood big display ────────────────────────────────
        panel(frame, 10, 65, 240, 160)
        txt(frame, "CURRENT MOOD", (20, 90), 0.48, GRAY)
        txt(frame, current_mood.upper(), (20, 145), 1.4, mood_color, bold=True)
        if USE_DEEPFACE and face_detected:
            txt(frame, f"Confidence: {mood_conf:.0f}%", (20, 175), 0.48, GRAY)
        elif not face_detected:
            txt(frame, "No face detected", (20, 175), 0.45, (0, 100, 220))

        # ── LEFT BOTTOM: dominant mood ────────────────────────────────────
        panel(frame, 10, 235, 240, 100)
        txt(frame, "DOMINANT TODAY", (20, 258), 0.48, GRAY)
        if mood_times:
            dominant = max(mood_times, key=mood_times.get)
            dom_color = MOOD_COLORS.get(dominant, WHITE)
            txt(frame, dominant.upper(), (20, 300), 1.0, dom_color, bold=True)
            txt(frame, fmt_time(mood_times[dominant]), (20, 325), 0.48, GRAY)

        # ── BOTTOM: hint ───────────────────────────────────────────────────
        panel(frame, 0, H-40, W, 40)
        txt(frame, "[Q] Quit & see session report     Stay in frame for accurate mood tracking",
            (20, H-14), 0.43, GRAY)

        # ── No face warning ────────────────────────────────────────────────
        if not face_detected:
            txt(frame, "Look at the camera!", (W//2-140, H//2), 0.9, (0, 180, 255), bold=True)

        cv2.imshow("MoodTime — AI Mood & Screen Time Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── SESSION REPORT ────────────────────────────────────────────────────────
    total = sum(mood_times.values())
    elapsed_total = time.time() - session_start

    print("\n" + "="*45)
    print("         MOODTIME SESSION REPORT")
    print("="*45)
    print(f"  Date          : {datetime.datetime.now().strftime('%d %B %Y')}")
    print(f"  Total Screen  : {fmt_time(elapsed_total)}")
    print(f"  Face Tracked  : {fmt_time(total)}")
    print("-"*45)
    print("  MOOD BREAKDOWN:")
    for mood, t in sorted(mood_times.items(), key=lambda x: -x[1]):
        pct = (t / max(total, 1)) * 100
        bar = "█" * int(pct / 5)
        print(f"  {mood.upper():<12} {bar:<20} {pct:5.1f}%  ({fmt_time(t)})")
    if mood_times:
        dominant = max(mood_times, key=mood_times.get)
        print("-"*45)
        print(f"  You were mostly : {dominant.upper()} today!")
    print("="*45)

if __name__ == "__main__":
    main()