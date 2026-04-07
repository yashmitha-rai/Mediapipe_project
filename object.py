import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog
import pyttsx3

# ---------------- COLOR DETECTION ----------------
def get_color_name(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_color = hsv.mean(axis=0).mean(axis=0)
    hue, sat, val = avg_color

    if val < 50:
        return "Black"
    if sat < 40 and val > 200:
        return "White"
    if sat < 40:
        return "Gray"

    if hue < 10 or hue > 170:
        return "Red"
    elif 10 <= hue < 25:
        return "Orange"
    elif 25 <= hue < 35:
        return "Yellow"
    elif 35 <= hue < 85:
        return "Green"
    elif 85 <= hue < 125:
        return "Blue"
    elif 125 <= hue < 150:
        return "Purple"
    else:
        return "Unknown"

# ---------------- VOICE ----------------
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ---------------- SETUP ----------------
root = Tk()
root.withdraw()

model = YOLO("yolov8n.pt")

while True:
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not image_path:
        break

    frame = cv2.imread(image_path)
    if frame is None:
        continue

    height, width, _ = frame.shape
    total_area = height * width

    results = model.predict(frame, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        speak("No objects detected")
    else:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            box_area = (x2 - x1) * (y2 - y1)
            area_percent = (box_area / total_area) * 100

            crop = frame[y1:y2, x1:x2]
            color = get_color_name(crop)

            label = f"{class_name} | {color} | {area_percent:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            sentence = f"{color} {class_name} detected occupying {int(area_percent)} percent of the image"
            speak(sentence)

    cv2.imshow("AI Voice Assistant Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Finished.")
