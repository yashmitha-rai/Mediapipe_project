import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

object_start_time = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    total_area = height * width

    results = model.predict(frame, verbose=False)
    result = results[0]

    current_objects = []

    if result.boxes is not None:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # -------- AREA CALCULATION --------
            box_area = (x2 - x1) * (y2 - y1)
            area_percent = (box_area / total_area) * 100

            # -------- DISTANCE ESTIMATION --------
            if area_percent > 30:
                distance = "Near"
            elif area_percent > 10:
                distance = "Medium"
            else:
                distance = "Far"

            current_objects.append(class_name)

            # -------- TIME TRACKING --------
            if class_name not in object_start_time:
                object_start_time[class_name] = time.time()

            visible_time = int(time.time() - object_start_time[class_name])

            color = (0, 255, 0)  # Always green now
            label = f"{class_name} | {distance} | {visible_time}s"

            # -------- DRAW --------
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color, 3)

            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    # -------- REMOVE OBJECTS WHEN DISAPPEAR --------
    for obj in list(object_start_time.keys()):
        if obj not in current_objects:
            del object_start_time[obj]

    cv2.imshow("VisionTrack AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
