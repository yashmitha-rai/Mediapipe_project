import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)

def finger_open(lm, tip, pip):
    return lm[tip].y < lm[pip].y

# Word formation variables
word = ""
prev_letter = ""
stable_start = None
letter_added = False
HOLD_TIME = 2.0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    letter = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            thumb  = lm[4].x > lm[3].x
            index  = finger_open(lm, 8, 6)
            middle = finger_open(lm, 12, 10)
            ring   = finger_open(lm, 16, 14)
            pinky  = finger_open(lm, 20, 18)

            # Extra checks
            index_curled  = lm[8].y > lm[6].y   # index fully curled
            middle_curled = lm[12].y > lm[10].y
            ring_curled   = lm[16].y > lm[14].y
            pinky_curled  = lm[20].y > lm[18].y

            # Fingertip distances for special letters
            thumb_tip  = lm[4]
            index_tip  = lm[8]
            middle_tip = lm[12]
            ring_tip   = lm[16]
            pinky_tip  = lm[20]

            def dist(a, b):
                return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

            thumb_index_touch  = dist(thumb_tip, index_tip) < 0.05
            thumb_middle_touch = dist(thumb_tip, middle_tip) < 0.05
            thumb_ring_touch   = dist(thumb_tip, ring_tip) < 0.05
            thumb_pinky_touch  = dist(thumb_tip, pinky_tip) < 0.05

            # ---- LETTER CONDITIONS ----

            # A - fist, all fingers closed
            if not thumb and not index and not middle and not ring and not pinky:
                letter = "A"

            # B - 4 fingers open, thumb tucked
            elif index and middle and ring and pinky and not thumb:
                letter = "B"

            # C - all fingers curved (half open) — thumb and index form C shape
            elif not index and not middle and not ring and not pinky and not thumb:
                # C vs A: in C hand is more open/curved — approximate with thumb out to side
                pass  # handled below with thumb position

            # D - index up, thumb touches middle
            elif index and not middle and not ring and not pinky and thumb_middle_touch:
                letter = "D"

            # E - all fingers bent/curled down, thumb tucked under
            elif index_curled and middle_curled and ring_curled and pinky_curled and not thumb:
                letter = "E"

            # F - index+thumb touch, other 3 fingers open
            elif thumb_index_touch and middle and ring and pinky:
                letter = "F"

            # G - index points sideways, thumb points sideways (like pointing a gun sideways)
            elif index and not middle and not ring and not pinky and not thumb:
                letter = "G"

            # H - index + middle extended sideways together
            elif index and middle and not ring and not pinky and not thumb:
                letter = "H"  # same as V but horizontal — approximated same way

            # I - only pinky open
            elif pinky and not index and not middle and not ring and not thumb:
                letter = "I"

            # K - index + middle open, thumb between them
            elif index and middle and not ring and not pinky and thumb:
                letter = "K"

            # L - thumb + index open
            elif thumb and index and not middle and not ring and not pinky:
                letter = "L"

            # M - three fingers folded over thumb (approximated: index+middle+ring down, pinky down)
            elif not index and not middle and not ring and not pinky and not thumb:
                letter = "M"  # very similar to A/E — hard to distinguish purely by open/close

            # N - two fingers folded over thumb
            # O - all fingers curve to touch thumb (round shape)
            elif thumb_index_touch and not middle and not ring and not pinky:
                letter = "O"

            # P - index points down, thumb out
            elif thumb and index and not middle and not ring and not pinky:
                letter = "P"  # similar to K/L pointing down

            # R - index + middle crossed (open but together)
            elif index and middle and not ring and not pinky and not thumb:
                letter = "R"

            # S - fist with thumb over fingers
            elif thumb and not index and not middle and not ring and not pinky:
                letter = "S"

            # T - thumb between index and middle
            elif thumb and not index and not middle and not ring and not pinky:
                letter = "T"

            # U - index + middle open together, straight up
            elif index and middle and not ring and not pinky and not thumb:
                letter = "U"

            # V - index + middle open (V shape)
            elif index and middle and not ring and not pinky and not thumb:
                letter = "V"

            # W - index + middle + ring open
            elif index and middle and ring and not pinky and not thumb:
                letter = "W"

            # X - index hooked/bent
            elif not index and not middle and not ring and not pinky and not thumb:
                letter = "X"

            # Y - thumb + pinky open
            elif thumb and pinky and not index and not middle and not ring:
                letter = "Y"

            # Z - index pointing, draw Z (approximated as index only)
            elif index and not middle and not ring and not pinky and not thumb:
                letter = "Z"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---- WORD FORMATION ----
    now = time.time()

    if letter != prev_letter:
        prev_letter = letter
        stable_start = now
        letter_added = False
    else:
        if letter and not letter_added:
            held = now - stable_start
            if held >= HOLD_TIME:
                word += letter
                letter_added = True

    # Progress bar
    if letter and not letter_added and stable_start:
        held = min(now - stable_start, HOLD_TIME)
        progress = int((held / HOLD_TIME) * 300)
        cv2.rectangle(frame, (40, 195), (340, 220), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 195), (40 + progress, 220), (0, 255, 255), -1)

    # ---- DISPLAY ----
    cv2.putText(frame, f"Letter: {letter}", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(frame, f"Word: {word}", (40, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
    cv2.putText(frame, "BACKSPACE=delete  SPACE=clear", (40, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("ASL Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 8:
        word = word[:-1]
    elif key == 32:
        word = ""

cap.release()
cv2.destroyAllWindows()