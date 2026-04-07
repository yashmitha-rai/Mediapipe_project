"""
Driver Drowsiness Detection - Simple & Reliable
================================================
Uses MediaPipe Face Mesh
- EAR (Eye Aspect Ratio) for eye closure
- Head pitch for nodding detection

Press Q to quit, C to calibrate EAR, R to reset

Install requirements:
    pip install opencv-python mediapipe numpy pygame sounddevice
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import threading
import struct
import wave
import io

# ── WAV generator ─────────────────────────────────────────────────────────────
def _generate_wav_bytes(freq=1000, duration=0.4):
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            val = int(32767 * np.sin(2 * np.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack('<h', val))
    buf.seek(0)
    return buf.read()


# ── Auto-detect best audio method at startup ──────────────────────────────────
_beep_method = None

def _detect_beep_method():
    global _beep_method

    # Method 1: sounddevice
    try:
        import sounddevice as sd
        sample_rate = 44100
        t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
        audio = (np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
        sd.play(audio, samplerate=sample_rate, blocking=True)
        _beep_method = "sounddevice"
        print("[Audio] Using: sounddevice ✓")
        return
    except Exception:
        pass

    # Method 2: pygame
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        sample_rate = 44100
        t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
        buf = (np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(buf)
        sound.play()
        time.sleep(0.15)
        _beep_method = "pygame"
        print("[Audio] Using: pygame ✓")
        return
    except Exception:
        pass

    # Method 3: winsound PlaySound WAV in memory
    try:
        import winsound
        wav_data = _generate_wav_bytes(1000, 0.1)
        winsound.PlaySound(wav_data, winsound.SND_MEMORY)
        _beep_method = "winsound_playsound"
        print("[Audio] Using: winsound.PlaySound ✓")
        return
    except Exception:
        pass

    # Method 4: winsound.Beep (PC speaker)
    try:
        import winsound
        winsound.Beep(1000, 100)
        _beep_method = "winsound_beep"
        print("[Audio] Using: winsound.Beep ✓")
        return
    except Exception:
        pass

    # Method 5: PowerShell
    try:
        import subprocess
        subprocess.call(
            ["powershell", "-c", "[console]::beep(1000,100)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _beep_method = "powershell"
        print("[Audio] Using: PowerShell beep ✓")
        return
    except Exception:
        pass

    _beep_method = "none"
    print("[Audio] WARNING: No audio method worked!")
    print("        Try: pip install sounddevice   OR   pip install pygame")


def beep(freq=1000, duration=0.4):
    """Play alert beep using the best available audio method."""
    def _play():
        try:
            if _beep_method == "sounddevice":
                import sounddevice as sd
                sample_rate = 44100
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
                sd.play(audio, samplerate=sample_rate, blocking=True)

            elif _beep_method == "pygame":
                import pygame
                sample_rate = 44100
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                buf = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(buf)
                sound.play()
                time.sleep(duration + 0.05)

            elif _beep_method == "winsound_playsound":
                import winsound
                wav_data = _generate_wav_bytes(freq, duration)
                winsound.PlaySound(wav_data, winsound.SND_MEMORY)

            elif _beep_method == "winsound_beep":
                import winsound
                winsound.Beep(max(37, min(32767, int(freq))), int(duration * 1000))

            elif _beep_method == "powershell":
                import subprocess
                subprocess.call(
                    ["powershell", "-c", f"[console]::beep({int(freq)},{int(duration*1000)})"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                print("\a", end="", flush=True)

        except Exception as e:
            print(f"\a[Beep error: {e}]", flush=True)

    threading.Thread(target=_play, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────

mp_face = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

GREEN  = (0, 200, 0)
YELLOW = (0, 200, 255)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (100, 100, 100)

def ear(lm, pts, W, H):
    def pt(i):
        return np.array([lm[pts[i]].x * W, lm[pts[i]].y * H])
    A = np.linalg.norm(pt(1) - pt(5))
    B = np.linalg.norm(pt(2) - pt(4))
    C = np.linalg.norm(pt(0) - pt(3))
    return (A + B) / (2.0 * C + 1e-6)

def get_head_pitch(lm):
    nose_y     = lm[1].y
    forehead_y = lm[10].y
    chin_y     = lm[152].y
    face_h     = abs(chin_y - forehead_y) + 1e-6
    center_y   = (forehead_y + chin_y) / 2.0
    return (nose_y - center_y) / face_h

def put_text(frame, text, pos, scale=0.7, color=WHITE, thickness=2):
    cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 1)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_eye(frame, lm, pts, W, H, color):
    pts_px = [(int(lm[p].x * W), int(lm[p].y * H)) for p in pts]
    for i in range(len(pts_px)):
        cv2.line(frame, pts_px[i], pts_px[(i + 1) % len(pts_px)], color, 1)

EAR_THRESH      = 0.20
EAR_CLOSED_SECS = 2.0
NOD_THRESH      = 0.08
NOD_SECS        = 1.5


def main():
    global EAR_THRESH

    print("Detecting audio output method...")
    _detect_beep_method()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    eye_closed_since = None
    nod_since        = None
    blink_count      = 0
    last_eye_open    = True
    session_start    = time.time()
    calib_msg        = ""
    calib_msg_time   = 0.0
    last_beep_time   = 0.0

    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            H, W  = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = fm.process(rgb)
            rgb.flags.writeable = True

            now            = time.time()
            avg_ear        = 0.30
            pitch          = 0.0
            face_found     = False
            eye_closed_dur = 0.0
            nod_dur        = 0.0
            status         = "AWAKE"
            status_col     = GREEN

            if res.multi_face_landmarks:
                face_found = True
                lm = res.multi_face_landmarks[0].landmark

                ear_l   = ear(lm, LEFT_EYE,  W, H)
                ear_r   = ear(lm, RIGHT_EYE, W, H)
                avg_ear = (ear_l + ear_r) / 2.0
                pitch   = get_head_pitch(lm)

                eyes_closed = avg_ear < EAR_THRESH
                if not eyes_closed and not last_eye_open:
                    blink_count += 1
                last_eye_open = not eyes_closed

                if eyes_closed:
                    if eye_closed_since is None:
                        eye_closed_since = now
                    eye_closed_dur = now - eye_closed_since
                else:
                    eye_closed_since = None
                    eye_closed_dur   = 0.0

                nodding = pitch > NOD_THRESH
                if nodding:
                    if nod_since is None:
                        nod_since = now
                    nod_dur = now - nod_since
                else:
                    nod_since = None
                    nod_dur   = 0.0

                if eye_closed_dur >= EAR_CLOSED_SECS:
                    status     = "DROWSY!"
                    status_col = RED
                elif nod_dur >= NOD_SECS:
                    status     = "NODDING!"
                    status_col = YELLOW
                elif eye_closed_dur >= 1.0:
                    status     = "WARNING"
                    status_col = YELLOW
                else:
                    status     = "AWAKE"
                    status_col = GREEN

                ec = RED if eyes_closed else GREEN
                draw_eye(frame, lm, LEFT_EYE,  W, H, ec)
                draw_eye(frame, lm, RIGHT_EYE, W, H, ec)

            # ── BEEP ALERT ────────────────────────────────────────────
            beep_interval = 1.5 if status == "DROWSY!" else 3.0
            if status in ("DROWSY!", "NODDING!", "WARNING"):
                if now - last_beep_time >= beep_interval:
                    freq = 1400 if status == "DROWSY!" else 900
                    beep(freq=freq, duration=0.4)
                    last_beep_time = now

            # DROWSY flash overlay
            if status == "DROWSY!":
                alpha = 0.15 + 0.10 * abs(np.sin(now * 5))
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 180), -1)
                cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

            # ── TOP BAR ───────────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (W, 55), (20, 20, 20), -1)
            put_text(frame, "DRIVER DROWSINESS DETECTION",
                     (10, 38), 0.8, (0, 220, 255), 2)
            elapsed = int(now - session_start)
            put_text(frame, f"{elapsed//60:02d}:{elapsed%60:02d}",
                     (W - 110, 38), 0.75, WHITE)

            # ── STATUS BOX ────────────────────────────────────────────
            sx, sy = W - 230, 65
            cv2.rectangle(frame, (sx - 10, sy - 10), (W - 5, sy + 105),
                          (25, 25, 25), -1)
            cv2.rectangle(frame, (sx - 10, sy - 10), (W - 5, sy + 105),
                          status_col, 2)
            put_text(frame, "STATUS",  (sx, sy + 22), 0.5,  GRAY)
            put_text(frame, status,    (sx, sy + 75), 1.1,  status_col, 3)

            # ── LEFT INFO PANEL ───────────────────────────────────────
            cv2.rectangle(frame, (5, 65), (265, 320), (20, 20, 20), -1)
            cv2.rectangle(frame, (5, 65), (265, 320), (70, 70, 70),  1)

            ear_col = RED if avg_ear < EAR_THRESH else GREEN
            put_text(frame, "EAR (Eye Ratio)",           (15, 95),  0.52, GRAY)
            put_text(frame, f"{avg_ear:.3f}",             (15, 128), 0.85, ear_col, 2)
            put_text(frame, f"/ thresh {EAR_THRESH:.2f}", (110, 128), 0.48, GRAY)

            bar_w    = 235
            bar_fill = int(min(avg_ear / 0.40, 1.0) * bar_w)
            cv2.rectangle(frame, (15, 138), (15 + bar_w, 152), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, 138), (15 + bar_fill, 152), ear_col,    -1)
            cv2.rectangle(frame, (15, 138), (15 + bar_w, 152),    WHITE,       1)
            tx = 15 + int((EAR_THRESH / 0.40) * bar_w)
            cv2.line(frame, (tx, 136), (tx, 154), (0, 220, 255), 2)

            if eye_closed_dur > 0:
                ec_txt = f"CLOSED  {eye_closed_dur:.1f}s"
                ec_col = RED if eye_closed_dur > 1.0 else YELLOW
            else:
                ec_txt = "OPEN"
                ec_col = GREEN
            put_text(frame, ec_txt, (15, 180), 0.6, ec_col, 2)

            p_col = YELLOW if pitch > NOD_THRESH else GREEN
            put_text(frame, "Head Pitch",      (15, 215), 0.52, GRAY)
            put_text(frame, f"{pitch:+.3f}",   (15, 245), 0.75, p_col, 2)
            if nod_dur > 0:
                put_text(frame, f"Nodding {nod_dur:.1f}s", (120, 245), 0.52, YELLOW)

            put_text(frame, f"Blinks: {blink_count}", (15, 285), 0.55, WHITE)

            # Audio method indicator
            put_text(frame, f"Audio: {_beep_method}", (15, H - 45), 0.40, GRAY, 1)

            if not face_found:
                put_text(frame, "NO FACE DETECTED — Move closer!",
                         (W//2 - 220, H//2), 0.9, RED, 3)

            if calib_msg and now - calib_msg_time < 3.0:
                put_text(frame, calib_msg,
                         (W//2 - 220, H - 50), 0.65, (0, 220, 255), 2)

            cv2.rectangle(frame, (0, H - 35), (W, H), (20, 20, 20), -1)
            put_text(frame,
                     "Q = Quit    C = Calibrate EAR (keep eyes open)    R = Reset",
                     (20, H - 10), 0.48, GRAY)

            cv2.imshow("Driver Drowsiness Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                eye_closed_since = None
                nod_since        = None
                blink_count      = 0
                session_start    = now
                calib_msg        = "Session reset!"
                calib_msg_time   = now
            elif key == ord('c'):
                if face_found and avg_ear > 0.15:
                    EAR_THRESH     = round(avg_ear * 0.75, 3)
                    calib_msg      = f"Calibrated! Threshold set to {EAR_THRESH:.3f}"
                    calib_msg_time = now
                else:
                    calib_msg      = "Keep eyes open and face visible to calibrate"
                    calib_msg_time = now

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended.  Blinks detected: {blink_count}")


if __name__ == "__main__":
    main()