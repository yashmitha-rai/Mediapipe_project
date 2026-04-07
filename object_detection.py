import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Hide Tkinter root window
root = Tk()
root.withdraw()

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")


def get_color_name(object_crop):
    # Convert to HSV
    hsv = cv2.cvtColor(object_crop, cv2.COLOR_BGR2HSV)

    # Remove very dark pixels (shadow removal)
    mask = hsv[:, :, 2] > 50
    hsv_filtered = hsv[mask]

    if len(hsv_filtered) == 0:
        return "Unknown"

    avg_hue = np.mean(hsv_filtered[:, 0])
    avg_sat = np.mean(hsv_filtered[:, 1])
    avg_val = np.mean(hsv_filtered[:, 2])

    # Detect black / white / gray
    if avg_val < 60:
        return "Black"
    if avg_sat < 40 and avg_val > 200:
        return "White"
    if avg_sat < 40:
        return "Gray"

    # Color detection by Hue
    if avg_hue < 10 or avg_hue > 170:
        return "Red"
    elif 10 <= avg_hue < 25:
        return "Orange"
    elif 25 <= avg_hue < 35:
        return "Yellow"
    elif 35 <= avg_hue < 85:
        return "Green"
    elif 85 <= avg_hue < 125:
        return "Blue"
    elif 125 <= avg_hue < 150:
        return "Purple"
    else:
        return "Unknown"


while True:
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not image_path:
        print("No image selected. Exiting...")
        break

    frame = cv2.imread(image_path)
    if frame is None:
        print("Error loading image!")
        continue

    height, width, _ = frame.shape
    image_area = height * width

    results = model.predict(frame, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected!")
    else:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            # -------- AREA CALCULATION --------
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            area_percent = (box_area / image_area) * 100

            # -------- COLOR CALCULATION --------
            object_crop = frame[y1:y2, x1:x2]
            color_name = get_color_name(object_crop)

            print("Detected:", class_name)
            print("Color:", color_name)
            print(f"Area: {area_percent:.2f}%")
            print("------------------")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Ensure text stays inside screen
            text_y = max(30, y1 - 40)

            cv2.putText(frame,
                        f"{class_name} ({confidence:.2f})",
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.putText(frame,
                        f"Color: {color_name}",
                        (x1, text_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

            cv2.putText(frame,
                        f"Area: {area_percent:.1f}%",
                        (x1, text_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

    cv2.imshow("YOLO Object Detection + Analysis", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    again = input("Do you want to detect another image? (y/n): ").lower()
    if again != 'y':
        break

print("Program finished.")
