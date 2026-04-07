import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
session_start = time.time()
total_bad_time = 0
alert_buffer = deque(maxlen=20)

# ---- AUTO CALIBRATION ----
calibrated = False
calib_data = []
calib_start = time.time()
base_neck_drop = 0
base_ear_dist = 0
base_shoulder_diff = 0

def pt(lm, i, w, h): return np.array([lm[i].x*w, lm[i].y*h])
def dist(a, b): return np.linalg.norm(a-b)
def panel(f,x,y,w,h):
    o=f.copy(); cv2.rectangle(o,(x,y),(x+w,y+h),(15,15,15),-1); cv2.addWeighted(o,.75,f,.25,0,f)
def bar(f,x,y,w,h,val,mx,col,lbl):
    cv2.rectangle(f,(x,y),(x+w,y+h),(40,40,40),-1)
    cv2.rectangle(f,(x,y),(x+int(w*min(val/mx,1)),y+h),col,-1)
    cv2.putText(f,lbl,(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1)

print("Posture Guard Started! Sit straight for 5 seconds to calibrate...")

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame,1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pr, fr = pose.process(rgb), face_mesh.process(rgb)
    now = time.time()
    neck_drop = shoulder_diff = face_dist = ear_dist = 0
    is_bad, alerts = False, []

    # ---- CALIBRATION SCREEN ----
    if not calibrated:
        elapsed_calib = now - calib_start
        remaining = max(0, 5 - int(elapsed_calib))

        if pr.pose_landmarks:
            lm = pr.pose_landmarks.landmark
            PL = mp_pose.PoseLandmark
            ls = pt(lm, PL.LEFT_SHOULDER, w, h)
            rs = pt(lm, PL.RIGHT_SHOULDER, w, h)
            le = pt(lm, PL.LEFT_EAR, w, h)
            re = pt(lm, PL.RIGHT_EAR, w, h)
            nose = pt(lm, PL.NOSE, w, h)
            smid = (ls+rs)/2
            calib_data.append({
                "neck": nose[1]-smid[1],
                "ear": (dist(le,ls)+dist(re,rs))/2,
                "shoulder": abs(ls[1]-rs[1])
            })

        # Calibration UI
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
        cv2.putText(frame,"SIT STRAIGHT & LOOK AT CAMERA",(w//2-220,h//2-60),
                    cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
        cv2.putText(frame,"Calibrating your posture...",(w//2-180,h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        cv2.putText(frame,f"Starting in {remaining} seconds...",(w//2-160,h//2+50),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,100),2)

        # Progress bar
        progress = min(elapsed_calib/5, 1.0)
        cv2.rectangle(frame,(w//2-200,h//2+90),(w//2+200,h//2+115),(40,40,40),-1)
        cv2.rectangle(frame,(w//2-200,h//2+90),(w//2-200+int(400*progress),h//2+115),(0,255,100),-1)

        if elapsed_calib >= 5 and len(calib_data) > 10:
            base_neck_drop    = np.mean([d["neck"] for d in calib_data])
            base_ear_dist     = np.mean([d["ear"] for d in calib_data])
            base_shoulder_diff = np.mean([d["shoulder"] for d in calib_data])
            calibrated = True
            print(f"Calibrated! neck={base_neck_drop:.1f} ear={base_ear_dist:.1f} shoulder={base_shoulder_diff:.1f}")

        cv2.imshow("Posture Guard", frame)
        if cv2.waitKey(1)&0xFF==27: break
        continue

    # ---- POSE DETECTION ----
    if pr.pose_landmarks:
        lm = pr.pose_landmarks.landmark
        PL = mp_pose.PoseLandmark
        ls, rs = pt(lm,PL.LEFT_SHOULDER,w,h), pt(lm,PL.RIGHT_SHOULDER,w,h)
        le, re = pt(lm,PL.LEFT_EAR,w,h),       pt(lm,PL.RIGHT_EAR,w,h)
        nose   = pt(lm,PL.NOSE,w,h)
        smid   = (ls+rs)/2

        neck_drop     = nose[1]-smid[1]
        shoulder_diff = abs(ls[1]-rs[1])
        ear_dist      = (dist(le,ls)+dist(re,rs))/2

        # Compare against calibrated baseline with tolerance
        if neck_drop     < base_neck_drop - 40:  alerts.append("HEAD DROPPING DOWN!");   is_bad=True
        if shoulder_diff > base_shoulder_diff+20: alerts.append("NECK TILTED SIDEWAYS!"); is_bad=True
        if ear_dist      < base_ear_dist - 30:    alerts.append("SLOUCHING!");            is_bad=True

        mp_draw.draw_landmarks(frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=3),
            mp_draw.DrawingSpec(color=(0,180,180),thickness=2))

    # ---- FACE MESH ----
    if fr.multi_face_landmarks:
        for fl in fr.multi_face_landmarks:
            flm = fl.landmark
            face_dist = dist(pt(flm,10,w,h), pt(flm,152,w,h))
            if face_dist > 280: alerts.append("TOO CLOSE TO SCREEN!"); is_bad=True
            mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_CONTOURS,
                None, mp_styles.get_default_face_mesh_contours_style())

    alert_buffer.append(is_bad)
    is_bad_smooth = sum(alert_buffer) > len(alert_buffer)*0.4
    if is_bad_smooth: total_bad_time += 0.03

    # ---- UI ----
    panel(frame,5,5,320,320)
    cv2.putText(frame,"POSTURE GUARD",(15,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    cv2.line(frame,(15,42),(310,42),(60,60,60),1)

    status,col,icon = ("BAD POSTURE",(0,0,255),"X") if is_bad_smooth else ("GOOD POSTURE",(0,255,100),"OK")
    cv2.putText(frame,f"{icon} {status}",(15,80),cv2.FONT_HERSHEY_SIMPLEX,1.1,col,3)
    cv2.line(frame,(15,100),(310,100),(60,60,60),1)

    bar(frame,15,125,280,18,max(base_neck_drop-neck_drop,0),50,(0,200,255),"Head Drop")
    bar(frame,15,163,280,18,max(shoulder_diff-base_shoulder_diff,0),30,(255,180,0),"Shoulder Tilt")
    bar(frame,15,201,280,18,max(base_ear_dist-ear_dist,0),40,(255,100,0),"Slouch Level")
    bar(frame,15,239,280,18,face_dist,300,(0,150,255),"Screen Distance")
    cv2.line(frame,(15,268),(310,268),(60,60,60),1)

    e=int(now-session_start); m,s=divmod(e,60)
    bm,bs=divmod(int(total_bad_time),60)
    cv2.putText(frame,f"Session     : {m:02d}:{s:02d}",(15,290),cv2.FONT_HERSHEY_SIMPLEX,.55,(200,200,200),1)
    cv2.putText(frame,f"Bad Posture : {bm:02d}:{bs:02d}",(15,313),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1)

    if alerts and is_bad_smooth:
        for i,a in enumerate(alerts):
            cv2.putText(frame,f"! {a}",(w-300,40+i*30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)

    cv2.putText(frame,"ESC = Quit",(w-120,h-15),cv2.FONT_HERSHEY_SIMPLEX,.5,(150,150,150),1)
    cv2.imshow("Posture Guard", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()