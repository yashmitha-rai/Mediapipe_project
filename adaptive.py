"""
Adaptive UI System
==================
DISTANCE EFFECTS:
  FAR   → screen brighter + text bigger + text color BRIGHT WHITE
  CLOSE → screen darker  + text smaller + text color DIM GRAY
  NORMAL→ screen normal  + text normal  + text color normal

Other detections:
  Close eyes 2s → RED alert + beep
  Turn head     → WARNING
  Nod down      → WARNING

Install: pip install opencv-python mediapipe numpy sounddevice
Press Q to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import collections
import io, wave, struct

# ── Beep ──────────────────────────────────────────────────────────────────────
def beep():
    def _p():
        try:
            import sounddevice as sd
            t = np.linspace(0, 0.4, 17640, endpoint=False)
            sd.play((np.sin(2*np.pi*1200*t)*32767).astype(np.int16), 44100, blocking=True)
        except:
            try:
                import winsound
                buf = io.BytesIO()
                with wave.open(buf,'wb') as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
                    for i in range(17640):
                        w.writeframes(struct.pack('<h',int(32767*np.sin(2*np.pi*1200*i/44100))))
                buf.seek(0); winsound.PlaySound(buf.read(), winsound.SND_MEMORY)
            except: pass
    threading.Thread(target=_p, daemon=True).start()

# ── Landmarks ─────────────────────────────────────────────────────────────────
L_EYE = [362,385,387,263,373,380]
R_EYE = [33,160,158,133,153,144]

def get_ear(lm, pts, W, H):
    p = lambda i: np.array([lm[pts[i]].x*W, lm[pts[i]].y*H])
    return (np.linalg.norm(p(1)-p(5))+np.linalg.norm(p(2)-p(4)))/(2*np.linalg.norm(p(0)-p(3))+1e-6)

def get_iod(lm, W, H):
    return np.linalg.norm([(lm[33].x-lm[263].x)*W,(lm[33].y-lm[263].y)*H])

def get_yaw(lm):
    fw = abs(lm[454].x-lm[234].x)+1e-6
    return (lm[1].x-(lm[234].x+lm[454].x)/2)/fw

def get_pitch(lm):
    fh = abs(lm[152].y-lm[10].y)+1e-6
    return (lm[1].y-(lm[10].y+lm[152].y)/2)/fh

class Smooth:
    def __init__(self, n=10): self.q = collections.deque(maxlen=n)
    def __call__(self, v):    self.q.append(v); return float(np.mean(self.q))

def txt(img, text, x, y, size, color, bold=False):
    th = 2 if bold else 1
    cv2.putText(img, text, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), th+1)
    cv2.putText(img, text, (x,y),     cv2.FONT_HERSHEY_SIMPLEX, size, color,   th)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sm_ear = Smooth(12); sm_iod = Smooth(15)
    sm_yaw = Smooth(10); sm_pitch = Smooth(10)

    eye_since = 0; blink_n = 0; last_open = True
    blink_log = collections.deque(maxlen=30)
    last_beep = 0; t0 = time.time()

    brightness = 1.0
    zoom       = 1.0
    # Font color components (smoothly animated)
    font_r = 220.0
    font_g = 220.0
    font_b = 220.0

    with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as fm:

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            H, W  = frame.shape[:2]
            now   = time.time()

            ear_val=0.28; iod=120.0; yaw=0.0; pitch=0.0
            closed_dur=0.0; face_found=False

            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                face_found = True
                lm = res.multi_face_landmarks[0].landmark
                ear_val = sm_ear((get_ear(lm,L_EYE,W,H)+get_ear(lm,R_EYE,W,H))/2)
                iod     = sm_iod(get_iod(lm,W,H))
                yaw     = sm_yaw(get_yaw(lm))
                pitch   = sm_pitch(get_pitch(lm))

                closed = ear_val < 0.20
                if not closed and not last_open:
                    blink_n += 1; blink_log.append(now)
                last_open = not closed
                if closed:
                    if not eye_since: eye_since = now
                    closed_dur = now - eye_since
                else: eye_since = 0

                ec = (0,50,220) if closed else (0,220,80)
                for pts in [L_EYE, R_EYE]:
                    px = [(int(lm[p].x*W),int(lm[p].y*H)) for p in pts]
                    for i in range(len(px)):
                        cv2.line(frame, px[i], px[(i+1)%len(px)], ec, 1)

            # ── Conditions ────────────────────────────────────────────
            bpm       = sum(1 for t in blink_log if now-t<60)
            fatigue   = closed_dur >= 2.0
            too_close = iod > 155
            too_far   = iod < 65
            head_turn = abs(yaw) > 0.18
            nodding   = pitch > 0.12

            if fatigue or bpm > 25:           severity = 2
            elif closed_dur>1 or too_close or \
                 too_far or head_turn or nodding or \
                 (bpm<8 and now-t0>30):       severity = 1
            else:                             severity = 0

            # ── Targets based on distance ─────────────────────────────
            if too_far:
                target_bright = 1.6    # screen gets bright
                target_zoom   = 1.6    # panel gets bigger
                # Font: BRIGHT WHITE — easy to read from far
                tr, tg, tb = 255.0, 255.0, 255.0

            elif too_close:
                target_bright = 0.35   # screen gets dark
                target_zoom   = 0.70   # panel gets smaller
                # Font: DIM GRAY — reduce eye strain when too close
                tr, tg, tb = 90.0, 90.0, 90.0

            elif fatigue:
                target_bright = 1.4
                target_zoom   = 1.3
                tr, tg, tb = 220.0, 220.0, 220.0

            else:
                target_bright = 1.0    # normal
                target_zoom   = 1.0    # normal
                # Font: normal soft white
                tr, tg, tb = 220.0, 220.0, 220.0

            # Smooth transitions
            brightness += (target_bright - brightness) * 0.12
            zoom       += (target_zoom   - zoom)       * 0.10
            font_r     += (tr - font_r) * 0.10
            font_g     += (tg - font_g) * 0.10
            font_b     += (tb - font_b) * 0.10

            brightness = float(np.clip(brightness, 0.2, 1.8))
            zoom       = float(np.clip(zoom, 0.5, 1.8))
            font_color = (int(font_b), int(font_g), int(font_r))  # BGR

            # ── Apply brightness to frame ──────────────────────────────
            display = np.clip(frame.astype(np.float32)*brightness, 0, 255).astype(np.uint8)

            # ── Apply color tint ───────────────────────────────────────
            if severity == 2:
                alpha = 0.18 + 0.10*abs(np.sin(now*4))
                tint  = display.copy()
                cv2.rectangle(tint, (0,0),(W,H),(0,0,200),-1)
                cv2.addWeighted(tint, alpha, display, 1-alpha, 0, display)
            elif severity == 1:
                tint = display.copy()
                cv2.rectangle(tint, (0,0),(W,H),(0,120,220),-1)
                cv2.addWeighted(tint, 0.10, display, 0.90, 0, display)

            # ── Info Panel (size & font color both change) ─────────────
            z   = zoom
            lh  = int(38*z)
            pw  = int(310*z)
            ph  = int(295*z)
            lx  = 22
            ly  = 22 + int(30*z)

            # Panel background
            ov = display.copy()
            cv2.rectangle(ov, (10,10),(10+pw, 10+ph),(15,15,15),-1)
            cv2.addWeighted(ov, 0.78, display, 0.22, 0, display)

            # Panel border color
            border = (0,60,220) if severity==2 else (0,160,220) if severity==1 else (0,180,80)
            cv2.rectangle(display,(10,10),(10+pw,10+ph), border, 2)

            # Title always bright cyan
            txt(display, "ADAPTIVE UI", lx, ly, 0.68*z, (0,220,255), bold=True)
            ly += lh

            # ── Distance label ─────────────────────────────────────────
            if too_far:
                d_label = "FAR  - bright + big text"
                d_col   = (255, 255, 255)   # white
            elif too_close:
                d_label = "CLOSE - dark + dim text"
                d_col   = (90, 90, 90)      # gray
            else:
                d_label = "Distance: NORMAL"
                d_col   = (0, 220, 80)
            txt(display, d_label, lx, ly, 0.50*z, d_col)
            ly += lh

            # ── Brightness bar ─────────────────────────────────────────
            txt(display, f"Brightness: {brightness:.2f}x", lx, ly, 0.50*z, font_color)
            bw = int(180*z)
            bf = int(np.clip(brightness/1.8,0,1)*bw)
            cv2.rectangle(display,(lx,ly+4),(lx+bw,ly+int(11*z)),(40,40,40),-1)
            cv2.rectangle(display,(lx,ly+4),(lx+bf, ly+int(11*z)),font_color,-1)
            ly += lh

            # ── Zoom bar ───────────────────────────────────────────────
            txt(display, f"Text zoom:  {zoom:.2f}x", lx, ly, 0.50*z, font_color)
            zf = int(np.clip(zoom/1.8,0,1)*bw)
            cv2.rectangle(display,(lx,ly+4),(lx+bw,ly+int(11*z)),(40,40,40),-1)
            cv2.rectangle(display,(lx,ly+4),(lx+zf, ly+int(11*z)),font_color,-1)
            ly += lh

            # ── Font color preview ─────────────────────────────────────
            txt(display, "Font color:", lx, ly, 0.50*z, (150,150,150))
            # Show a color swatch
            cv2.rectangle(display,
                          (lx+int(115*z), ly-int(14*z)),
                          (lx+int(190*z), ly+int(4*z)),
                          font_color, -1)
            if too_far:
                txt(display, "BRIGHT", lx+int(195*z), ly, 0.44*z, (255,255,255))
            elif too_close:
                txt(display, "DIM", lx+int(195*z), ly, 0.44*z, (90,90,90))
            else:
                txt(display, "NORMAL", lx+int(195*z), ly, 0.44*z, (0,200,80))
            ly += lh

            # ── Eyes ───────────────────────────────────────────────────
            if closed_dur > 0:
                e_txt = f"Eyes: CLOSED {closed_dur:.1f}s"
                e_col = (0,40,255) if closed_dur>1 else (0,180,255)
            else:
                e_txt, e_col = "Eyes: OPEN", font_color
            txt(display, e_txt, lx, ly, 0.50*z, e_col)
            ly += lh

            # ── Head ───────────────────────────────────────────────────
            if head_turn:
                h_txt = f"Head: {'LEFT' if yaw<0 else 'RIGHT'}"
                h_col = (0,180,255)
            elif nodding:
                h_txt, h_col = "Head: NODDING", (0,180,255)
            else:
                h_txt, h_col = "Head: FOCUSED", font_color
            txt(display, h_txt, lx, ly, 0.50*z, h_col)
            ly += lh

            # ── Blinks ─────────────────────────────────────────────────
            b_col = (0,40,255) if bpm>25 else (0,180,255) if (bpm<8 and now-t0>30) else font_color
            txt(display, f"Blinks: {bpm}/min", lx, ly, 0.50*z, b_col)

            # ── Big centre status ──────────────────────────────────────
            s_txt = "! ALERT !" if severity==2 else "WARNING" if severity==1 else "NORMAL"
            s_col = (0,40,255)  if severity==2 else (0,180,255) if severity==1 else (0,220,80)
            tw = cv2.getTextSize(s_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0][0]
            txt(display, s_txt, W//2-tw//2, 52, 1.2, s_col, bold=True)

            # ── No face ────────────────────────────────────────────────
            if not face_found:
                txt(display, "NO FACE — MOVE CLOSER", W//2-200, H//2, 1.0, (0,40,255), bold=True)

            # ── Bottom hint ────────────────────────────────────────────
            cv2.rectangle(display,(0,H-30),(W,H),(10,10,10),-1)
            txt(display, "Q = Quit   |   Move face CLOSE = dark+dim    Move face FAR = bright+white",
                12, H-8, 0.44, (120,120,120))

            # ── Beep ───────────────────────────────────────────────────
            if severity==2 and now-last_beep>2.0:
                beep(); last_beep=now

            cv2.imshow("Adaptive UI System", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()