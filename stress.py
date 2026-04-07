import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
session_start = time.time()

stress_history = deque(maxlen=60)
brow_buffer    = deque(maxlen=15)
mouth_buffer   = deque(maxlen=15)
eye_buffer     = deque(maxlen=15)
nose_buffer    = deque(maxlen=15)
alert_timer    = 0
avg_ear        = 0

calibrated  = False
calib_data  = []
calib_start = time.time()
base_brow   = 0
base_mouth  = 0
base_ear    = 0
base_nose   = 0

TIPS = [
    "Take a deep breath!",
    "Close your eyes for 10s",
    "Drink some water",
    "Stretch your neck",
    "Look away from screen",
]

LEFT_EYE   = [362,385,387,263,373,380]
RIGHT_EYE  = [33,160,158,133,153,144]
LEFT_BROW  = [336,296,334,293,300]
RIGHT_BROW = [107,66,105,63,70]

def pt(lm,i,w,h): return np.array([lm[i].x*w, lm[i].y*h])
def dist(a,b): return np.linalg.norm(a-b)
def panel(f,x,y,w,h):
    o=f.copy(); cv2.rectangle(o,(x,y),(x+w,y+h),(15,15,15),-1); cv2.addWeighted(o,.75,f,.25,0,f)
def bar(f,x,y,w,h,val,mx,col,lbl):
    cv2.rectangle(f,(x,y),(x+w,y+h),(40,40,40),-1)
    cv2.rectangle(f,(x,y),(x+int(w*min(abs(val)/max(mx,0.001),1)),y+h),col,-1)
    if lbl: cv2.putText(f,lbl,(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1)
def get_ear(lm,indices,w,h):
    pts=[pt(lm,i,w,h) for i in indices]
    return (dist(pts[1],pts[5])+dist(pts[2],pts[4]))/(2.0*max(dist(pts[0],pts[3]),0.001))

print("Stress Detector — Sit relaxed for 5 seconds to calibrate!")

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame,1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_res = face_mesh.process(rgb)
    now = time.time()

    if not calibrated:
        elapsed   = now - calib_start
        remaining = max(0, 5-int(elapsed))

        if face_res.multi_face_landmarks:
            for fl in face_res.multi_face_landmarks:
                lm = fl.landmark
                fh = dist(pt(lm,10,w,h), pt(lm,152,w,h))
                lec = np.mean([pt(lm,i,w,h) for i in LEFT_EYE],  axis=0)
                rec = np.mean([pt(lm,i,w,h) for i in RIGHT_EYE], axis=0)
                lbc = np.mean([pt(lm,i,w,h) for i in LEFT_BROW],  axis=0)
                rbc = np.mean([pt(lm,i,w,h) for i in RIGHT_BROW], axis=0)
                brow    = (dist(lec,lbc)+dist(rec,rbc))/(2*max(fh,0.001))
                ul      = pt(lm,13,w,h); ll = pt(lm,14,w,h)
                mouth   = dist(ul,ll)/max(fh,0.001)
                ear_val = (get_ear(lm,LEFT_EYE,w,h)+get_ear(lm,RIGHT_EYE,w,h))/2
                ln      = pt(lm,114,w,h); rn = pt(lm,343,w,h)
                nose_w  = dist(ln,rn)/max(fh,0.001)
                calib_data.append({"brow":brow,"mouth":mouth,"ear":ear_val,"nose":nose_w})

        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
        cv2.putText(frame,"SIT RELAXED & LOOK AT CAMERA",(w//2-210,h//2-60),
                    cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
        cv2.putText(frame,"Calibrating your baseline...",(w//2-170,h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        cv2.putText(frame,f"Starting in {remaining} seconds...",(w//2-160,h//2+50),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,100),2)
        progress = min(elapsed/5,1.0)
        cv2.rectangle(frame,(w//2-200,h//2+90),(w//2+200,h//2+115),(40,40,40),-1)
        cv2.rectangle(frame,(w//2-200,h//2+90),(w//2-200+int(400*progress),h//2+115),(0,255,100),-1)

        if elapsed >= 5 and len(calib_data) > 10:
            base_brow  = np.mean([d["brow"]  for d in calib_data])
            base_mouth = np.mean([d["mouth"] for d in calib_data])
            base_ear   = np.mean([d["ear"]   for d in calib_data])
            base_nose  = np.mean([d["nose"]  for d in calib_data])
            calibrated = True
            session_start = time.time()
            print(f"Calibrated! brow={base_brow:.3f} mouth={base_mouth:.3f} ear={base_ear:.3f} nose={base_nose:.3f}")

        cv2.imshow("Stress Detector", frame)
        if cv2.waitKey(1)&0xFF==27: break
        continue

    eye_strain = brow_furrow = mouth_tension = nose_wrinkle = 0

    if face_res.multi_face_landmarks:
        for fl in face_res.multi_face_landmarks:
            lm = fl.landmark
            fh = dist(pt(lm,10,w,h), pt(lm,152,w,h))

            lev     = get_ear(lm,LEFT_EYE,w,h)
            rev     = get_ear(lm,RIGHT_EYE,w,h)
            avg_ear = (lev+rev)/2
            eye_strain = max((avg_ear-base_ear)/max(base_ear,0.001), 0)
            eye_buffer.append(eye_strain)

            lec  = np.mean([pt(lm,i,w,h) for i in LEFT_EYE],  axis=0)
            rec  = np.mean([pt(lm,i,w,h) for i in RIGHT_EYE], axis=0)
            lbc  = np.mean([pt(lm,i,w,h) for i in LEFT_BROW],  axis=0)
            rbc  = np.mean([pt(lm,i,w,h) for i in RIGHT_BROW], axis=0)
            brow = (dist(lec,lbc)+dist(rec,rbc))/(2*max(fh,0.001))
            brow_furrow = max((base_brow-brow)/max(base_brow,0.001), 0)
            brow_buffer.append(brow_furrow)

            ul = pt(lm,13,w,h); ll = pt(lm,14,w,h)
            mouth = dist(ul,ll)/max(fh,0.001)
            mouth_tension = max((base_mouth-mouth)/max(base_mouth,0.001), 0)
            mouth_buffer.append(mouth_tension)

            ln = pt(lm,114,w,h); rn = pt(lm,343,w,h)
            nose_ratio = dist(ln,rn)/max(fh,0.001)
            nose_wrinkle = max((base_nose-nose_ratio)/max(base_nose,0.001), 0)
            nose_buffer.append(nose_wrinkle)

            mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_CONTOURS,
                None, mp_styles.get_default_face_mesh_contours_style())

    eye_avg   = np.mean(list(eye_buffer))   if eye_buffer   else 0
    brow_avg  = np.mean(list(brow_buffer))  if brow_buffer  else 0
    mouth_avg = np.mean(list(mouth_buffer)) if mouth_buffer else 0
    nose_avg  = np.mean(list(nose_buffer))  if nose_buffer  else 0

    stress_score = int(min((
        brow_avg  * 0.35 +
        mouth_avg * 0.25 +
        eye_avg   * 0.20 +
        nose_avg  * 0.20
    ) * 300, 100))

    stress_history.append(stress_score)

    if stress_score >= 70:   level,lcol = "HIGH STRESS",(0,0,255)
    elif stress_score >= 40: level,lcol = "MODERATE",   (0,200,255)
    else:                    level,lcol = "RELAXED",     (0,255,100)

    # ---- ALERT BOTTOM ----
    if stress_score >= 70: alert_timer = 80
    if alert_timer > 0:
        alert_timer -= 1
        tip = TIPS[int(now)%len(TIPS)]
        cv2.rectangle(frame,(5,h-80),(500,h-40),(20,20,100),-1)
        cv2.rectangle(frame,(5,h-80),(500,h-40),(0,0,255),2)
        cv2.putText(frame,f"! STRESS ALERT  Tip: {tip}",(10,h-53),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)

    # ---- STRESS GRAPH ----
    if len(stress_history) > 1:
        gx,gy,gw,gh = w-165,h-110,150,70
        panel(frame,gx-5,gy-20,gw+10,gh+28)
        cv2.putText(frame,"Stress History",(gx,gy-5),cv2.FONT_HERSHEY_SIMPLEX,.4,(200,200,200),1)
        pts = list(stress_history)
        for i in range(1,len(pts)):
            x1=gx+int((i-1)*gw/len(pts)); x2=gx+int(i*gw/len(pts))
            y1=gy+gh-int(pts[i-1]*gh/100); y2=gy+gh-int(pts[i]*gh/100)
            col=(0,0,255) if pts[i]>=70 else (0,200,255) if pts[i]>=40 else (0,255,100)
            cv2.line(frame,(x1,y1),(x2,y2),col,1)

    # ---- UI PANEL ----
    panel(frame,5,5,320,320)
    cv2.putText(frame,"STRESS DETECTOR",(15,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    cv2.line(frame,(15,42),(310,42),(60,60,60),1)
    cv2.putText(frame,f"{stress_score}%",(15,95),cv2.FONT_HERSHEY_SIMPLEX,1.8,lcol,3)
    cv2.putText(frame,level,(130,90),cv2.FONT_HERSHEY_SIMPLEX,.8,lcol,2)
    cv2.line(frame,(15,110),(310,110),(60,60,60),1)

    bar(frame,15,130,280,14,brow_avg, 0.1,(255,180,0), "Eyebrow Furrow")
    bar(frame,15,163,280,14,mouth_avg,0.1,(255,100,0), "Mouth Tension")
    bar(frame,15,196,280,14,eye_avg,  0.1,(0,200,255), "Eye Strain")
    bar(frame,15,229,280,14,nose_avg, 0.1,(0,180,255), "Nose Wrinkle")
    cv2.line(frame,(15,255),(310,255),(60,60,60),1)

    e=int(now-session_start); m,s=divmod(e,60)
    cv2.putText(frame,f"Session : {m:02d}:{s:02d}",(15,275),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1)
    cv2.putText(frame,f"brow:{brow_avg:.4f} mouth:{mouth_avg:.4f}",(15,295),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)
    cv2.putText(frame,f"eye:{eye_avg:.4f}  nose:{nose_avg:.4f}",(15,315),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)

    if alert_timer <= 0:
        cv2.putText(frame,"ESC=Quit | R=Recalibrate",(10,h-15),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(150,150,150),1)

    cv2.imshow("Stress Detector", frame)

    key = cv2.waitKey(1)&0xFF
    if key==27: break
    if key==ord('r'):
        calibrated=False; calib_data=[]; calib_start=time.time()
        stress_history.clear(); brow_buffer.clear()
        mouth_buffer.clear(); eye_buffer.clear()
        nose_buffer.clear()

cap.release()
cv2.destroyAllWindows()