import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils
styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
mouth_history, open_events, speaking_buffer = deque(maxlen=30), deque(maxlen=50), deque(maxlen=20)
was_open = False
mouth_open_count = 0
session_start = speaking_start = time.time()

def pt(lm, i, w, h): return np.array([lm[i].x * w, lm[i].y * h])
def dist(a, b): return np.linalg.norm(a - b)
def panel(f, x, y, w, h):
    o = f.copy(); cv2.rectangle(o,(x,y),(x+w,y+h),(15,15,15),-1); cv2.addWeighted(o,.75,f,.25,0,f)
def bar(f, x, y, w, h, val, mx, col, lbl):
    cv2.rectangle(f,(x,y),(x+w,y+h),(40,40,40),-1)
    cv2.rectangle(f,(x,y),(x+int(w*min(val/mx,1)),y+h),col,-1)
    cv2.putText(f,lbl,(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1)

print("Lip Sync Detector Started! Press ESC to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    now = time.time()
    is_speaking, mouth_ratio, attention = False, 0, 0

    if res.multi_face_landmarks:
        for fl in res.multi_face_landmarks:
            lm = fl.landmark
            ul, ll = pt(lm,13,w,h), pt(lm,14,w,h)
            fh = dist(pt(lm,10,w,h), pt(lm,152,w,h))
            mouth_ratio = dist(ul,ll)/fh if fh>0 else 0
            mouth_history.append(mouth_ratio)
            mouth_open = mouth_ratio > 0.025

            if mouth_open and not was_open:
                open_events.append(now); mouth_open_count += 1
            was_open = mouth_open

            speaking_buffer.append(len([t for t in open_events if now-t<1.5])>=3)
            is_speaking = sum(speaking_buffer) > len(speaking_buffer)*0.4

            # Attention Score
            eye_open = min((abs(lm[159].y - lm[145].y) + abs(lm[386].y - lm[374].y)) / 2 / 0.02, 1.0)
            centered = 1.0 - abs(lm[1].x - 0.5) * 2
            tilt = abs(lm[33].y - lm[263].y) * 10
            straight = max(1.0 - tilt, 0)
            attention = int((eye_open * 0.4 + centered * 0.3 + straight * 0.3) * 100)

            draw.draw_landmarks(frame, fl, mp.solutions.face_mesh.FACEMESH_LIPS,
                None, styles.get_default_face_mesh_contours_style())
            for p in [ul, ll]:
                cv2.circle(frame,(int(p[0]),int(p[1])),4,(0,255,255),-1)
            cv2.line(frame,(int(ul[0]),int(ul[1])),(int(ll[0]),int(ll[1])),(0,255,255),1)

    panel(frame, 5, 5, 310, 280)
    cv2.putText(frame,"LIP SYNC DETECTOR",(15,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    cv2.line(frame,(15,42),(300,42),(60,60,60),1)

    status,col,icon = ("SPEAKING",(0,255,100),"●") if is_speaking else ("SILENT",(100,100,100),"○")
    cv2.putText(frame,f"{icon} {status}",(15,80),cv2.FONT_HERSHEY_SIMPLEX,1.2,col,3)
    cv2.line(frame,(15,100),(300,100),(60,60,60),1)

    bar(frame,15,125,280,18,mouth_ratio,.08,(0,200,255),"Mouth Opening")
    bar(frame,15,168,280,18,len([t for t in open_events if now-t<1.5]),8,(0,255,100),"Speaking Activity")
    cv2.line(frame,(15,200),(300,200),(60,60,60),1)

    e = int(now-session_start); m,s = divmod(e,60)

    # Attention color
    att_col = (0,255,100) if attention>=80 else (0,200,255) if attention>=50 else (0,0,255)

    for txt,y in [
        (f"Session     : {m:02d}:{s:02d}", 225),
        (f"Mouth opens : {mouth_open_count}", 248),
        (f"Attention   : {attention}%", 271)]:
        cv2.putText(frame,txt,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(200,200,200),1)

    # Attention color indicator
    cv2.putText(frame,f"{attention}%",(220,271),cv2.FONT_HERSHEY_SIMPLEX,.55,att_col,2)

    cv2.putText(frame,"ESC = Quit",(w-120,h-15),cv2.FONT_HERSHEY_SIMPLEX,.5,(150,150,150),1)
    cv2.imshow("Lip Sync Detector", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()