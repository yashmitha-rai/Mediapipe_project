import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Tip IDs: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
TIP_IDS = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # 1. Draw original landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. Get all tip coordinates
            tips = []
            for tip_id in TIP_IDS:
                lm = hand_landmarks.landmark[tip_id]
                tips.append((int(lm.x * w), int(lm.y * h)))

            # 3. Calculate distance between adjacent tips
            # Loop through tips: (4,8), (8,12), (12,16), (16,20)
            for i in range(len(tips) - 1):
                p1 = tips[i]
                p2 = tips[i+1]
                
                # Pythagorean Theorem
                dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                
                # Draw a line between tips
                cv2.line(image, p1, p2, (0, 255, 255), 2)
                
                # Calculate midpoint to display distance text
                mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                cv2.putText(image, str(int(dist)), (mid_x, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 4. Display Hand Label
            label = handedness.classification[0].label
            wrist = hand_landmarks.landmark[0]
            cv2.putText(image, label, (int(wrist.x * w), int(wrist.y * h) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Multi-Finger Distance', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()