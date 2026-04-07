import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame

# ── SETUP ──
pygame.mixer.init()
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)
pose      = mp_pose.Pose(min_detection_confidence=0.5)

L_IRIS=[474,475,476,477]; R_IRIS=[469,470,471,472]
L_EYE=[33,133];           R_EYE=[362,263]

score=50.0; in_t=out_t=0.0; last_alert=0
heatmap=None; show_heat=False; start=time.time()

def beep():
    t=np.linspace(0,0.3,13230)
    w=(np.sin(2*np.pi*880*t)*32767).astype(np.int16)
    pygame.sndarray.make_sound(np.column_stack([w,w])).play()

def gaze(lm,fw,fh):
    li=np.mean([[lm[i].x*fw,lm[i].y*fh] for i in L_IRIS],axis=0)
    ri=np.mean([[lm[i].x*fw,lm[i].y*fh] for i in R_IRIS],axis=0)
    lw=abs(lm[L_EYE[1]].x-lm[L_EYE[0]].x)*fw
    rw=abs(lm[R_EYE[1]].x-lm[R_EYE[0]].x)*fw
    avg=((li[0]-lm[L_EYE[0]].x*fw)/(lw+1e-6)+(ri[0]-lm[R_EYE[0]].x*fw)/(rw+1e-6))/2
    gx,gy=int((li[0]+ri[0])/2),int((li[1]+ri[1])/2)
    return gx,gy,("Center" if 0.38<avg<0.62 else "Right" if avg<0.38 else "Left")

cap=cv2.VideoCapture(0); cap.set(3,960); cap.set(4,540)
prev=time.time()

while True:
    ret,frame=cap.read()
    if not ret: break
    frame=cv2.flip(frame,1); fh,fw=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    dt=time.time()-prev; prev=time.time()

    zx1,zy1,zx2,zy2=int(fw*0.15),int(fh*0.10),int(fw*0.85),int(fh*0.90)

    # Pose
    pr=pose.process(rgb); inside=False
    if pr.pose_landmarks:
        mp_draw.draw_landmarks(frame,pr.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec((100,200,255),2,3),mp_draw.DrawingSpec((60,80,140),2))
        n=pr.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nx,ny=int(n.x*fw),int(n.y*fh)
        inside=zx1<nx<zx2 and zy1<ny<zy2

    if inside: in_t+=dt
    else:
        out_t+=dt
        if time.time()-last_alert>3: beep(); last_alert=time.time()

    # Gaze
    gx,gy,gdir=fw//2,fh//2,"Center"
    fr=face_mesh.process(rgb)
    if fr.multi_face_landmarks:
        try:
            lm=fr.multi_face_landmarks[0].landmark
            gx,gy,gdir=gaze(lm,fw,fh)
            if heatmap is None: heatmap=np.zeros((fh,fw),np.float32)
            cv2.circle(heatmap,(gx,gy),40,1.0,-1)
            cv2.circle(frame,(gx,gy),10,(0,220,255),-1)
            cv2.putText(frame,gdir,(gx+15,gy+5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,220,255),1)
        except: pass

    rs=(100 if inside else 0)*0.6+(100 if gdir=="Center" else 40)*0.4
    score=0.95*score+0.05*rs

    # Heatmap
    if show_heat and heatmap is not None:
        hn=cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX)
        hb=cv2.GaussianBlur(hn.astype(np.uint8),(61,61),0)
        hc=cv2.applyColorMap(hb,cv2.COLORMAP_JET)
        mask=hb>10; frame[mask]=cv2.addWeighted(frame,0.4,hc,0.6,0)[mask]

    # Zone brackets
    col=(0,220,120) if inside else (0,0,255)
    for p1,p2 in [((zx1,zy1),(zx1+30,zy1)),((zx1,zy1),(zx1,zy1+30)),
                  ((zx2,zy1),(zx2-30,zy1)),((zx2,zy1),(zx2,zy1+30)),
                  ((zx1,zy2),(zx1+30,zy2)),((zx1,zy2),(zx1,zy2-30)),
                  ((zx2,zy2),(zx2-30,zy2)),((zx2,zy2),(zx2,zy2-30))]:
        cv2.line(frame,p1,p2,col,2)
    cv2.putText(frame,"IN ZONE ✅" if inside else "⚠ OUT OF ZONE",(zx1+5,zy1-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)

    if not inside:
        ov=frame.copy(); cv2.rectangle(ov,(0,0),(fw,fh),(0,0,180),-1)
        cv2.addWeighted(ov,0.18,frame,0.82,0,frame)

    # Left panel
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(200,fh),(12,12,22),-1)
    cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
    sc_col=(0,220,120) if score>=70 else (0,180,255) if score>=40 else (0,0,255)
    cv2.putText(frame,"FOCUS",(65,30),cv2.FONT_HERSHEY_DUPLEX,0.6,(160,120,255),1)
    cx,cy,r=100,110,60
    cv2.circle(frame,(cx,cy),r,(40,40,55),10)
    for a in range(0,int(score/100*360),3):
        rad=math.radians(a-90)
        cv2.circle(frame,(int(cx+r*math.cos(rad)),int(cy+r*math.sin(rad))),5,sc_col,-1)
    cv2.putText(frame,f"{int(score)}",(cx-20,cy+10),cv2.FONT_HERSHEY_DUPLEX,1.1,(255,255,255),2)
    grade="FOCUSED" if score>=70 else "MODERATE" if score>=40 else "DISTRACTED"
    cv2.putText(frame,grade,(10,190),cv2.FONT_HERSHEY_SIMPLEX,0.48,sc_col,1)
    cv2.putText(frame,f"In:  {int(in_t)}s",(10,220),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,220,120),1)
    cv2.putText(frame,f"Out: {int(out_t)}s",(10,242),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,80,255),1)
    cv2.putText(frame,f"Gaze:{gdir}",(10,264),cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,220,255),1)
    elapsed=int(time.time()-start)
    cv2.putText(frame,f"{elapsed//60:02d}:{elapsed%60:02d}",(10,fh-20),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,130),1)
    cv2.putText(frame,"H=Heat R=Reset ESC=End",(fw-220,fh-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(80,80,100),1)

    cv2.imshow("Focus Zone Tracker",frame)
    key=cv2.waitKey(1)&0xFF
    if key==27: break
    elif key==ord('h'): show_heat=not show_heat
    elif key==ord('r'): heatmap=None; in_t=out_t=0; score=50

cap.release(); cv2.destroyAllWindows()
print(f"\n── SESSION SUMMARY ──")
print(f"Focus Score : {int(score)}/100")
print(f"Grade       : {'FOCUSED' if score>=70 else 'MODERATE' if score>=40 else 'DISTRACTED'}")
print(f"In Zone     : {int(in_t)}s | Out Zone: {int(out_t)}s")
print(f"Duration    : {int(time.time()-start)}s")