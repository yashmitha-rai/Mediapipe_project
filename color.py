import cv2
import numpy as np

COLORS = {
    "Red":    ([(0,50,50),   (15,255,255)],  (0,0,255)),
    "Red2":   ([(165,50,50), (180,255,255)], (0,0,200)),
    "Orange": ([(16,50,50),  (25,255,255)],  (0,140,255)),
    "Yellow": ([(26,50,50),  (35,255,255)],  (0,255,255)),
    "Green":  ([(36,40,40),  (86,255,255)],  (0,255,0)),
    "Cyan":   ([(87,40,40),  (96,255,255)],  (255,255,0)),
    "Blue":   ([(97,40,40),  (130,255,255)], (255,0,0)),
    "Purple": ([(131,30,30), (155,255,255)], (255,0,180)),
    "Pink":   ([(156,30,30), (164,255,255)], (180,0,255)),
}

def panel(f,x,y,w,h):
    o=f.copy(); cv2.rectangle(o,(x,y),(x+w,y+h),(15,15,15),-1); cv2.addWeighted(o,.75,f,.25,0,f)

def bar(f,x,y,w,h,val,mx,col):
    cv2.rectangle(f,(x,y),(x+w,y+h),(40,40,40),-1)
    cv2.rectangle(f,(x,y),(x+int(w*min(val/mx,1)),y+h),col,-1)

cap = cv2.VideoCapture(0)
print("Color Detector Started! Press ESC to quit.")

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # Skin tone mask — exclude skin from detection
    skin_mask = cv2.inRange(hsv, np.array([0,20,70]), np.array([20,150,255]))
    skin_mask = cv2.dilate(skin_mask, None, iterations=3)

    detected = []

    for color_name, (hsv_range, bgr) in COLORS.items():
        mask = cv2.inRange(hsv,
               np.array(hsv_range[0]),
               np.array(hsv_range[1]))
        mask = cv2.erode(mask,  None, iterations=1)
        # Remove skin tone from mask
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt  = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Very low threshold — detects small objects too
        if area < 800:
            continue

        x1,y1,bw,bh = cv2.boundingRect(cnt)
        cx, cy = x1+bw//2, y1+bh//2
        pct = round((area/(w*h))*100, 1)
        display_name = "Red" if color_name == "Red2" else color_name

        cv2.rectangle(frame,(x1,y1),(x1+bw,y1+bh),bgr,2)
        label = f"{display_name} {pct}%"
        lw,lh = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.6,2)[0]
        cv2.rectangle(frame,(x1,y1-lh-10),(x1+lw+8,y1),bgr,-1)
        cv2.putText(frame,label,(x1+4,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        cv2.circle(frame,(cx,cy),5,bgr,-1)
        cv2.circle(frame,(cx,cy),8,bgr,2)
        detected.append((display_name, pct, cx, cy, bgr))

    # ---- UI PANEL ----
    panel_h = max(60 + len(detected)*40, 80)
    panel(frame, 5, 5, 310, panel_h)
    cv2.putText(frame,"COLOR DETECTOR",(15,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    cv2.line(frame,(15,42),(310,42),(60,60,60),1)

    if detected:
        seen, unique = set(), []
        for d in detected:
            if d[0] not in seen:
                seen.add(d[0]); unique.append(d)

        for i,(name,pct,cx,cy,bgr) in enumerate(unique):
            y = 70 + i*40
            cv2.circle(frame,(25,y-5),8,bgr,-1)
            cv2.putText(frame,f"{name}",(40,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.putText(frame,f"{pct}% at ({cx},{cy})",(130,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,180,180),1)
            bar(frame,15,y+8,280,8,pct,20,bgr)
    else:
        cv2.putText(frame,"Show a colored object...",(15,70),cv2.FONT_HERSHEY_SIMPLEX,.6,(100,100,100),1)

    total = len(set([d[0] for d in detected]))
    cv2.putText(frame,f"Colors detected: {total}",(w-210,h-15),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,200,255),1)
    cv2.putText(frame,"ESC = Quit",(w-120,30),cv2.FONT_HERSHEY_SIMPLEX,.5,(150,150,150),1)

    cv2.imshow("Color Detector", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()