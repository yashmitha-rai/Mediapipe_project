import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pickle
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
GESTURES = ["Hello", "Yes", "No", "ThankYou", "Sorry"]
SAMPLES  = 100
os.makedirs("data", exist_ok=True)

def panel(f,x,y,w,h):
    o=f.copy(); cv2.rectangle(o,(x,y),(x+w,y+h),(15,15,15),-1); cv2.addWeighted(o,.75,f,.25,0,f)

def bar(f,x,y,w,h,val,mx,col):
    cv2.rectangle(f,(x,y),(x+w,y+h),(40,40,40),-1)
    cv2.rectangle(f,(x,y),(x+int(w*min(val/mx,1)),y+h),col,-1)

# =====================
# PHASE 1 — COLLECT
# =====================
def collect():
    current, count, collecting = 0, 0, False
    csv_file = open("data/gestures.csv","w",newline="")
    writer   = csv.writer(csv_file)
    print("PHASE 1: Data Collection Started!")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame,1)
        h, w  = frame.shape[:2]
        res   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                lm = hl.landmark
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,200),thickness=2,circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,180,140),thickness=2))

                if collecting and current < len(GESTURES):
                    row = [current] + [v for l in lm for v in [l.x,l.y,l.z]]
                    writer.writerow(row)
                    count += 1
                    if count >= SAMPLES:
                        print(f"Done: {GESTURES[current]}")
                        current += 1; count = 0; collecting = False
                        if current >= len(GESTURES):
                            print("All gestures collected!")
                            csv_file.close()
                            return

        panel(frame,0,0,w,75)
        if current < len(GESTURES):
            cv2.putText(frame,f"Gesture: {GESTURES[current]}",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            col = (0,255,100) if collecting else (0,200,255)
            cv2.putText(frame,f"{'COLLECTING...' if collecting else 'Press SPACE to start'} ({count}/{SAMPLES})",(15,60),cv2.FONT_HERSHEY_SIMPLEX,.6,col,2)
            bar(frame,15,h-35,w-30,15,count,SAMPLES,(0,255,100))

        for i,g in enumerate(GESTURES):
            col = (0,255,100) if i<current else (0,200,255) if i==current else (100,100,100)
            cv2.putText(frame,f"{'✓' if i<current else '→' if i==current else '○'} {g}",(w-150,30+i*28),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1)

        cv2.imshow("Sign Language", frame)
        key = cv2.waitKey(1)&0xFF
        if key==27: csv_file.close(); return
        if key==32: collecting=True

# =====================
# PHASE 2 — TRAIN
# =====================
def train():
    print("PHASE 2: Training model...")
    X, y = [], []
    with open("data/gestures.csv","r") as f:
        for row in csv.reader(f):
            y.append(int(row[0]))
            X.append([float(v) for v in row[1:]])

    X, y = np.array(X), np.array(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    acc = accuracy_score(y_test,model.predict(X_test))
    print(f"Accuracy: {acc*100:.2f}%")
    with open("data/model.pkl","wb") as f:
        pickle.dump(model,f)
    print("Model saved!")
    return model

# =====================
# PHASE 3 — DETECT
# =====================
def detect(model):
    print("PHASE 3: Detection Started!")
    pred_buffer = deque(maxlen=15)
    sentence, last_word, last_time = [], "", 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame,1)
        h, w  = frame.shape[:2]
        res   = hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        prediction, confidence = "", 0

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                lm = hl.landmark
                mp_draw.draw_landmarks(frame,hl,mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,200),thickness=2,circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,180,140),thickness=2))

                row = [v for l in lm for v in [l.x,l.y,l.z]]
                proba = model.predict_proba([row])[0]
                confidence = max(proba)
                pred_buffer.append(np.argmax(proba))

                if len(pred_buffer)==15 and pred_buffer.count(max(set(pred_buffer),key=pred_buffer.count))>=10:
                    prediction = GESTURES[max(set(pred_buffer),key=pred_buffer.count)]

                now = time.time()
                if prediction and prediction!=last_word and now-last_time>2:
                    sentence.append(prediction)
                    last_word=prediction; last_time=now
                    if len(sentence)>5: sentence.pop(0)
        else:
            pred_buffer.clear(); last_word=""

        panel(frame,5,5,320,200)
        cv2.putText(frame,"SIGN LANGUAGE DETECTOR",(15,32),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        cv2.line(frame,(15,42),(310,42),(60,60,60),1)
        

        if prediction:
            cv2.putText(frame,prediction,(15,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,100),3)
            bar(frame,15,115,295,18,confidence,1,(0,255,100))
            cv2.putText(frame,f"Confidence: {int(confidence*100)}%",(15,150),cv2.FONT_HERSHEY_SIMPLEX,.55,(200,200,200),1)
        else:
            cv2.putText(frame,"Show a gesture...",(15,100),cv2.FONT_HERSHEY_SIMPLEX,.8,(100,100,100),2)

        cv2.line(frame,(15,165),(310,165),(60,60,60),1)
        cv2.putText(frame,f"Sentence: {' '.join(sentence)}",(15,190),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,200,255),1)

        panel(frame,w-160,5,155,175)
        cv2.putText(frame,"Gestures:",(w-150,30),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        for i,g in enumerate(GESTURES):
            cv2.putText(frame,g,(w-150,60+i*25),cv2.FONT_HERSHEY_SIMPLEX,.5,
                        (0,255,100) if g==prediction else (150,150,150),1)

        cv2.putText(frame,"C=Clear | ESC=Quit",(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,.45,(150,150,150),1)
        cv2.imshow("Sign Language",frame)

        key = cv2.waitKey(1)&0xFF
        if key==27: break
        if key==ord('c'): sentence.clear()

# =====================
# MAIN — RUN ALL
# =====================
MODEL_PATH = "data/model.pkl"

if os.path.exists(MODEL_PATH):
    print("Model found! Skipping to detection...")
    with open(MODEL_PATH,"rb") as f:
        model = pickle.load(f)
else:
    collect()
    model = train()

detect(model)

cap.release()
cv2.destroyAllWindows()