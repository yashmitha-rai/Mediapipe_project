import cv2
import mediapipe as mp
import numpy as np
import time
import threading

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

pose          = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# ─────────────────────────────────────────────
#  AGE & GENDER MODELS
# ─────────────────────────────────────────────
# Download & place in same folder:
# age_deploy.prototxt      → https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt
# age_net.caffemodel       → https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel
# gender_deploy.prototxt   → https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt
# gender_net.caffemodel    → https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_VALUES  = [1, 5, 10, 17, 28, 40, 50, 70]
GENDER_LIST = ['Male', 'Female']

age_net = gender_net = None
try:
    age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
    print("[INFO] Age model loaded!")
except: print("[WARN] Age model not found. Press M for manual input.")

try:
    gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
    print("[INFO] Gender model loaded!")
except: print("[WARN] Gender model not found. Press G for manual input.")

# ─────────────────────────────────────────────
#  EXERCISE DATABASE (Age + Gender)
# ─────────────────────────────────────────────
EXERCISES = {
    "teen": {
        "range": (10, 19), "label": "Teen (10-19)", "color": (0, 220, 120),
        "male":   ["Sprint Intervals - 4x30 sec", "Push Ups - 3x20 reps",
                   "Pull Ups - 3x10 reps", "Jump Squats - 3x15 reps", "Burpees - 3x10 reps"],
        "female": ["Jump Rope - 4x30 sec", "Knee Push Ups - 3x15 reps",
                   "Hip Bridges - 3x20 reps", "Lunges - 3x15 reps", "Yoga Stretching - 15 min"],
        "tip_male": "Build strength & explosiveness!",
        "tip_female": "Build flexibility & endurance!"
    },
    "young_adult": {
        "range": (20, 35), "label": "Young Adult (20-35)", "color": (0, 180, 255),
        "male":   ["Deadlifts - 4x12 reps", "Bench Press - 4x10 reps",
                   "Squats - 4x15 reps", "HIIT Cardio - 20 min", "Core Planks - 3x60 sec"],
        "female": ["Hip Thrusts - 4x12 reps", "Dumbbell Squats - 4x15 reps",
                   "Pilates Core - 20 min", "Resistance Band Glutes - 3x15", "HIIT Cardio - 15 min"],
        "tip_male": "Peak fitness — push your limits!",
        "tip_female": "Tone & strengthen your body!"
    },
    "middle_aged": {
        "range": (36, 55), "label": "Middle Aged (36-55)", "color": (0, 200, 255),
        "male":   ["Brisk Walking - 30 min", "Light Dumbbell Curls - 3x12",
                   "Swimming - 20 min", "Cycling - 20 min", "Lunges - 3x10 reps"],
        "female": ["Brisk Walking - 30 min", "Yoga - 20 min",
                   "Light Pilates - 20 min", "Swimming - 20 min", "Stretching - 15 min"],
        "tip_male": "Focus on endurance & joint health!",
        "tip_female": "Focus on flexibility & bone health!"
    },
    "senior": {
        "range": (56, 100), "label": "Senior (56+)", "color": (255, 180, 0),
        "male":   ["Gentle Walking - 20 min", "Chair Squats - 2x10",
                   "Resistance Band - 2x10", "Seated Leg Raises - 2x10", "Balance Exercises - 10 min"],
        "female": ["Gentle Walking - 20 min", "Chair Yoga - 15 min",
                   "Light Stretching - 15 min", "Seated Arm Raises - 2x10", "Water Aerobics - 20 min"],
        "tip_male": "Focus on balance & strength!",
        "tip_female": "Focus on flexibility & gentle movement!"
    }
}

def get_age_group(age):
    for key, val in EXERCISES.items():
        if val["range"][0] <= age <= val["range"][1]:
            return key
    return "young_adult"

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
detected_age     = None
detected_gender  = None
detecting        = False
last_detect_time = 0
DETECT_INTERVAL  = 4.0
manual_age_input = ""
input_mode       = False   # 'age' | 'gender' | False

# ─────────────────────────────────────────────
#  DETECTION THREAD
# ─────────────────────────────────────────────
def detect_age_gender(frame, x, y, w, h):
    global detected_age, detected_gender, detecting
    try:
        face_img = frame[max(0,y):y+h, max(0,x):x+w]
        if face_img.size == 0:
            detecting = False; return
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227,227),
            (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        if age_net:
            age_net.setInput(blob)
            preds = age_net.forward()
            detected_age = AGE_VALUES[preds[0].argmax()]
            print(f"[INFO] Age: {AGE_BUCKETS[preds[0].argmax()]}")
        if gender_net:
            gender_net.setInput(blob)
            preds = gender_net.forward()
            detected_gender = GENDER_LIST[preds[0].argmax()]
            print(f"[INFO] Gender: {detected_gender}")
    except Exception as e:
        print(f"[WARN] {e}")
    detecting = False

# ─────────────────────────────────────────────
#  DRAW FUNCTIONS
# ─────────────────────────────────────────────
scan_y = 0

def draw_scan_effect(frame):
    global scan_y
    fh, fw = frame.shape[:2]
    scan_y = (scan_y + 5) % fh
    cv2.line(frame, (0, scan_y), (fw-320, scan_y), (0,220,255), 1)
    ov = frame.copy()
    cv2.rectangle(ov, (0, scan_y-20), (fw-320, scan_y), (0,80,100), -1)
    cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

def draw_exercise_panel(frame, age, gender, group_key):
    fh, fw = frame.shape[:2]
    group  = EXERCISES[group_key]
    col    = group["color"]
    g      = (gender or "male").lower()
    exlist = group.get(g, group["male"])
    tip    = group.get(f"tip_{g}", "")
    g_icon  = "Male" if g == "male" else "Female"
    g_color = (180,150,255) if g == "male" else (255,120,200)

    ov = frame.copy()
    cv2.rectangle(ov, (fw-320, 0), (fw, fh), (12,12,22), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "EXERCISE PLAN",    (fw-305, 30), cv2.FONT_HERSHEY_DUPLEX,  0.62, col, 1)
    cv2.putText(frame, f"Age : {age}",     (fw-305, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,220), 1)
    cv2.putText(frame, f"Gender: {g_icon}",(fw-305, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, g_color, 1)
    cv2.putText(frame, group["label"],     (fw-305, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
    cv2.line(frame, (fw-315,106), (fw-5,106), (50,50,70), 1)

    for i, ex in enumerate(exlist):
        cv2.putText(frame, ex, (fw-310, 132+i*46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220,220,240), 1)

    cv2.line(frame, (fw-315,362), (fw-5,362), (50,50,70), 1)
    cv2.putText(frame, "Tip: "+tip, (fw-310,382),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,100), 1)

    bar_col = (160,100,255) if g == "male" else (255,80,180)
    cv2.rectangle(frame, (fw-320, fh-6), (fw, fh), bar_col, -1)

def draw_top_bar(frame, age, gender, scanning):
    fh, fw = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (fw-320,50), (12,12,22), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "FitAge - Exercise Recommender",
                (10,32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (160,120,255), 1)
    if scanning:
        cv2.putText(frame, "Scanning...", (fw-490,32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,255), 1)
    elif age or gender:
        info = f"Age:{age or '?'}  Gender:{gender or '?'}"
        cv2.putText(frame, info, (fw-490,32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,220,120), 1)

def draw_waiting_panel(frame):
    fh, fw = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (fw-320,0), (fw,fh), (12,12,22), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, "EXERCISE PLAN",    (fw-305,35),    cv2.FONT_HERSHEY_DUPLEX,  0.62, (80,80,120), 1)
    cv2.putText(frame, "Face camera to",   (fw-280,fh//2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,180), 1)
    cv2.putText(frame, "detect age & gender",(fw-295,fh//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150,150,180), 1)
    cv2.putText(frame, "or press M / G",   (fw-260,fh//2+38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,140), 1)

def draw_input_overlay(frame, text, mode):
    fh, fw = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (fw//2-220,fh//2-70), (fw//2+220,fh//2+80), (20,20,35), -1)
    cv2.addWeighted(ov, 0.92, frame, 0.08, 0, frame)
    cv2.rectangle(frame, (fw//2-220,fh//2-70), (fw//2+220,fh//2+80), (100,80,200), 2)
    if mode == 'age':
        cv2.putText(frame, "Enter Age:", (fw//2-100,fh//2-25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200,180,255), 1)
        cv2.putText(frame, text+"_",     (fw//2-30, fh//2+25), cv2.FONT_HERSHEY_DUPLEX, 1.1,  (255,255,255), 2)
    else:
        cv2.putText(frame, "Select Gender:", (fw//2-130,fh//2-25), cv2.FONT_HERSHEY_DUPLEX,  0.6, (200,180,255), 1)
        cv2.putText(frame, "1 = Male   2 = Female", (fw//2-160,fh//2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200,200,255), 1)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    global detected_age, detected_gender, detecting, last_detect_time
    global manual_age_input, input_mode, scan_y

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    WIN = "FitAge - Exercise Recommender"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)
    current_group = "young_adult"

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── POSE ──
        pose_res = pose.process(rgb)
        col = EXERCISES[current_group]["color"]
        if pose_res.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=col, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(60,60,100), thickness=2))

        # ── FACE ──
        face_res = face_detector.process(rgb)
        face_box = None
        if face_res.detections:
            det = face_res.detections[0]
            bb  = det.location_data.relative_bounding_box
            x1,y1 = int(bb.xmin*fw), int(bb.ymin*fh)
            bw,bh = int(bb.width*fw), int(bb.height*fh)
            face_box = (x1,y1,bw,bh)
            g = (detected_gender or "").lower()
            fc = (160,100,255) if g=="male" else (255,80,180) if g=="female" else col
            cv2.rectangle(frame, (x1,y1), (x1+bw,y1+bh), fc, 2)
            lbl = ""
            if detected_age:    lbl += f"Age~{detected_age} "
            if detected_gender: lbl += detected_gender
            if lbl: cv2.putText(frame, lbl, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fc, 1)

        # ── AUTO DETECT ──
        now = time.time()
        if (not detecting and face_box and
                (now-last_detect_time) > DETECT_INTERVAL and (age_net or gender_net)):
            detecting = True; last_detect_time = now
            x1,y1,bw,bh = face_box
            threading.Thread(target=detect_age_gender,
                             args=(frame.copy(),x1,y1,bw,bh), daemon=True).start()

        if detected_age: current_group = get_age_group(detected_age)
        if detecting: draw_scan_effect(frame)

        draw_top_bar(frame, detected_age, detected_gender, detecting)
        if detected_age or detected_gender:
            draw_exercise_panel(frame, detected_age, detected_gender, current_group)
        else:
            draw_waiting_panel(frame)

        if input_mode: draw_input_overlay(frame, manual_age_input, input_mode)

        cv2.rectangle(frame, (0,fh-30), (fw-320,fh), (12,12,22), -1)
        cv2.putText(frame, "R=Rescan  M=Manual Age  G=Manual Gender  ESC=Exit",
                    (10,fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80,80,100), 1)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        if input_mode == 'age':
            if key == 13:
                if manual_age_input.isdigit():
                    detected_age = int(manual_age_input)
                    current_group = get_age_group(detected_age)
                input_mode = False; manual_age_input = ""
            elif key == 8: manual_age_input = manual_age_input[:-1]
            elif 48 <= key <= 57:
                if len(manual_age_input) < 3: manual_age_input += chr(key)
        elif input_mode == 'gender':
            if key == ord('1'):   detected_gender = 'Male';   input_mode = False
            elif key == ord('2'): detected_gender = 'Female'; input_mode = False
            elif key == 27:       input_mode = False
        else:
            if key == 27: break
            elif key == ord('r'): detected_age = None; detected_gender = None; last_detect_time = 0
            elif key == ord('m'): input_mode = 'age';    manual_age_input = ""
            elif key == ord('g'): input_mode = 'gender'

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("[INFO] Starting FitAge with Gender Detection!")
    print("[INFO] R=Rescan | M=Manual Age | G=Manual Gender | ESC=Exit")
    run()