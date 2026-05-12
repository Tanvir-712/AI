import cv2
import csv
import mediapipe as mp
import time

# =========================
# MEDIAPIPE SETUP
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# =========================
# LABEL INPUT
# =========================

label = input("Enter Label (A-Z): ").strip().upper()

if not label:
    print("Invalid label")
    exit()

# =========================
# CSV FILE
# =========================

file = open("dataset.csv", "a", newline="")
writer = csv.writer(file)

# =========================
# FEATURE EXTRACTION
# =========================

def extract_landmarks(hand_landmarks):

    x_ = []
    y_ = []

    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)

    min_x = min(x_)
    min_y = min(y_)

    data = []

    for lm in hand_landmarks.landmark:
        data.append(lm.x - min_x)
        data.append(lm.y - min_y)
        data.append(lm.z)

    return data

# =========================
# SAMPLING CONTROL (3/sec)
# =========================

SAMPLES_PER_SEC = 22
INTERVAL = 1.0 / SAMPLES_PER_SEC
last_save_time = 0

count = 0

print("\nPress Q to stop...\n")

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    detected = False

    # =========================
    # HAND DETECTION
    # =========================

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            detected = True

            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # SAVE ONLY 3 SAMPLES/SEC
            # =========================

            current_time = time.time()

            if current_time - last_save_time >= INTERVAL:

                data = extract_landmarks(hand_landmarks)

                writer.writerow(data + [label])

                count += 1

                last_save_time = current_time

    # =========================
    # UI
    # =========================

    cv2.putText(
        frame,
        f"Label: {label}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    status = "Hand Detected" if detected else "Show Hand"

    color = (0,255,0) if detected else (0,0,255)

    cv2.putText(
        frame,
        status,
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        "Rate: 5 samples/sec",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow("Collect Data", frame)

    # =========================
    # EXIT
    # =========================

    key = cv2.waitKey(1)

    if key in [ord('q'), ord('Q'), 27]:
        break

# =========================
# CLEANUP
# =========================

cap.release()
file.close()
cv2.destroyAllWindows()

print(f"\nSaved {count} samples for label [{label}]")