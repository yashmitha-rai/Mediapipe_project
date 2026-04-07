import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time

# ─────────────────────────────────────────────
#  TEXT TO SPEECH SETUP
# ─────────────────────────────────────────────
engine = pyttsx3.init()
engine.setProperty('rate', 160)   # speed
engine.setProperty('volume', 1.0)

speech_lock = threading.Lock()
last_spoken = {}          # track last time each message was spoken
SPEAK_COOLDOWN = 3.0      # seconds between same message

def speak(text, priority=False):
    """Speak text in a separate thread to avoid blocking."""
    now = time.time()
    with speech_lock:
        last_time = last_spoken.get(text, 0)
        if not priority and (now - last_time) < SPEAK_COOLDOWN:
            return
        last_spoken[text] = now

    def _speak():
        engine.say(text)
        engine.runAndWait()

    t = threading.Thread(target=_speak, daemon=True)
    t.start()

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_pose      = mp.solutions.pose
mp_face      = mp.solutions.face_detection
mp_hands     = mp.solutions.hands
mp_draw      = mp.solutions.drawing_utils
mp_obj       = mp.solutions.objectron

pose          = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# ─────────────────────────────────────────────
#  ZONE DEFINITIONS
# ─────────────────────────────────────────────
# Frame divided into 3 horizontal zones
def get_zone(x, frame_width):
    if x < frame_width * 0.33:
        return "left"
    elif x < frame_width * 0.66:
        return "center"
    else:
        return "right"

# ─────────────────────────────────────────────
#  DISTANCE ESTIMATION (by bounding box size)
# ─────────────────────────────────────────────
def estimate_distance(box_area, frame_area):
    ratio = box_area / frame_area
    if ratio > 0.35:   return "very close", (0, 0, 255)     # RED
    elif ratio > 0.15: return "close",      (0, 140, 255)   # ORANGE
    elif ratio > 0.05: return "nearby",     (0, 220, 255)   # YELLOW
    else:              return "far",        (0, 220, 120)   # GREEN

# ─────────────────────────────────────────────
#  DRAW ZONE LINES
# ─────────────────────────────────────────────
def draw_zones(frame):
    h, w = frame.shape[:2]
    cv2.line(frame, (w//3, 0),     (w//3, h),     (60,60,80), 1)
    cv2.line(frame, (2*w//3, 0),   (2*w//3, h),   (60,60,80), 1)
    cv2.putText(frame, "LEFT",   (20,       25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,120), 1)
    cv2.putText(frame, "CENTER", (w//3+50,  25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,120), 1)
    cv2.putText(frame, "RIGHT",  (2*w//3+20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,120), 1)

# ─────────────────────────────────────────────
#  DRAW DARK PANEL
# ─────────────────────────────────────────────
def draw_panel(frame, detections):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-160), (w, h), (10,10,20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "🦯 BLIND ASSISTANCE SYSTEM",
                (10, h-135), cv2.FONT_HERSHEY_DUPLEX, 0.6, (160,120,255), 1)

    if not detections:
        cv2.putText(frame, "✅ Clear path ahead",
                    (10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,120), 2)
    else:
        for i, d in enumerate(detections[:3]):
            cv2.putText(frame, f"⚠ {d}",
                        (10, h-100 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,140,255) if "very close" not in d else (0,0,255), 1)

    cv2.putText(frame, "Press ESC to exit",
                (w-160, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,100), 1)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    WIN = "Blind Assistance System"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)

    speak("Blind Assistance System started. Camera is active.", priority=True)

    clear_path_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections      = []
        audio_messages  = []

        # ── FACE DETECTION ──
        face_res = face_detector.process(rgb)
        if face_res.detections:
            for det in face_res.detections:
                bb  = det.location_data.relative_bounding_box
                x1  = int(bb.xmin * w)
                y1  = int(bb.ymin * h)
                bw  = int(bb.width * w)
                bh  = int(bb.height * h)
                x2, y2 = x1+bw, y1+bh
                cx  = x1 + bw//2

                area     = bw * bh
                dist, col = estimate_distance(area, w*h)
                zone     = get_zone(cx, w)

                # Draw box
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                label = f"Person | {dist} | {zone}"
                cv2.putText(frame, label, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                detections.append(f"Person {dist} on {zone}")

                # Audio
                if dist == "very close":
                    audio_messages.append(f"Warning! Person very close on your {zone}")
                elif dist == "close":
                    audio_messages.append(f"Person close on your {zone}")
                else:
                    audio_messages.append(f"Person detected on {zone}")

        # ── POSE DETECTION (body parts / obstacles) ──
        pose_res = pose.process(rgb)
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks.landmark

            # Check key body landmarks visible = someone is there
            nose = lms[mp_pose.PoseLandmark.NOSE]
            if nose.visibility > 0.5:
                nx = int(nose.x * w)
                ny = int(nose.y * h)
                zone = get_zone(nx, w)

                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame,
                    pose_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(100,200,255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(60,120,180), thickness=2)
                )

        # ── OBSTACLE SIMULATION via contour size ──
        # Detects large blobs (furniture, walls, objects)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (21,21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8000 or area > w*h*0.85:   # ignore tiny & full-frame
                continue
            x,y,cw,ch = cv2.boundingRect(cnt)
            cx = x + cw//2
            cy = y + ch//2

            # Skip if already covered by face detection
            already = any(abs(cx - w//2) < 50 for _ in detections)

            dist, col = estimate_distance(area, w*h)
            zone      = get_zone(cx, w)

            if dist in ("very close", "close"):
                cv2.rectangle(frame, (x,y), (x+cw,y+ch), (0,80,200), 1)
                cv2.putText(frame, f"Obstacle|{dist}|{zone}",
                            (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,80,200), 1)
                if f"Obstacle {dist} on {zone}" not in detections:
                    detections.append(f"Obstacle {dist} on {zone}")
                    audio_messages.append(f"Obstacle {dist} on your {zone}")

        # ── CLEAR PATH ──
        if not detections:
            if time.time() - clear_path_timer > 5.0:
                speak("Clear path ahead. Safe to move forward.")
                clear_path_timer = time.time()

        # ── SPEAK DETECTIONS ──
        for msg in audio_messages[:2]:   # max 2 at a time
            speak(msg)

        # ── DRAW UI ──
        draw_zones(frame)
        draw_panel(frame, detections)

        # Warning flash if very close
        if any("very close" in d for d in detections):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,180), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.putText(frame, "⚠ DANGER - STOP!",
                        (w//2-150, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)

        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    speak("System shutting down. Goodbye.", priority=True)
    time.sleep(1.5)
    cap.release()
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")