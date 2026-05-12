import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import pyttsx3

from collections import Counter
from tensorflow.keras.models import load_model

# =========================================================
# LOAD MODEL + LABELS
# =========================================================

model = load_model("model/sign_model.h5")

labels = np.load("model/labels.npy", allow_pickle=True)
labels = list(labels)

# =========================================================
# TEXT TO SPEECH
# =========================================================

speech_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 145)
    engine.setProperty("volume", 1.0)

    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.say(str(text))
            engine.runAndWait()
        except Exception as e:
            print("TTS Error:", e)
        finally:
            speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    if text and text.strip():
        speech_queue.put(text.strip())

# =========================================================
# MEDIAPIPE HANDS
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

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

    return np.array(data).reshape(1, -1)

# =========================================================
# VARIABLES
# =========================================================

buffer = []

sentence = ""
current_word = ""

predicted = ""
last_letter = ""

confidence = 0

last_add_time = 0
last_hand_time = time.time()

LETTER_DELAY = 1.0
SPACE_DELAY = 2.0

CONFIDENCE_THRESHOLD = 0.80

BUFFER_SIZE = 15

LETTER_LOCK = False

hand_gone_time = None
DOUBLE_LETTER_RESET = 0.5

print("\nAI Sign Language Translator Running...")
print("Press Q to Quit\n")

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    now = time.time()

    predicted = ""
    confidence = 0
    hand_visible = False

    # =====================================================
    # HAND DETECTION
    # =====================================================

    if result.multi_hand_landmarks:

        hand_visible = True
        last_hand_time = now

        if hand_gone_time is not None:
            gone_duration = now - hand_gone_time
            if gone_duration >= DOUBLE_LETTER_RESET:
                LETTER_LOCK = False
                last_letter = ""
            hand_gone_time = None

        predictions = []

        for hand_landmarks in result.multi_hand_landmarks:

            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            data = extract_landmarks(hand_landmarks)

            probs = model.predict(data, verbose=0)[0]

            idx = np.argmax(probs)
            conf = probs[idx]

            if conf >= CONFIDENCE_THRESHOLD:
                predictions.append(labels[idx])
                confidence = conf

        if predictions:
            buffer.append(Counter(predictions).most_common(1)[0][0])

    else:
        if hand_gone_time is None:
            hand_gone_time = now

    # =====================================================
    # SMOOTHING
    # =====================================================

    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    if buffer:
        predicted = Counter(buffer).most_common(1)[0][0]

    # =====================================================
    # LETTER INPUT LOGIC
    # =====================================================

    if predicted:

        if predicted != last_letter:
            last_letter = predicted
            last_add_time = now
            LETTER_LOCK = False

        elif not LETTER_LOCK:
            hold_time = now - last_add_time
            if hold_time >= LETTER_DELAY:
                current_word += predicted
                LETTER_LOCK = True

    else:
        LETTER_LOCK = False

    # =====================================================
    # AUTO SPACE + AUTO SPEAK EACH WORD
    # =====================================================

    if not hand_visible:

        idle_time = now - last_hand_time

        if idle_time >= SPACE_DELAY and current_word:

            word_to_speak = current_word

            sentence += current_word + " "
            current_word = ""
            last_letter = ""
            buffer.clear()
            LETTER_LOCK = False

            speak(word_to_speak)   # speaks every word, forever

    # =====================================================
    # KEYBOARD CONTROLS
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('q'), ord('Q'), 27]:
        break

    elif key == 13:
        full = (sentence + current_word).strip()
        if full:
            speak(full)

    elif key == 8:
        current_word = current_word[:-1]

    elif key == ord(' '):
        if current_word:
            sentence += current_word + " "
            current_word = ""

    elif key in [ord('c'), ord('C')]:
        sentence = ""
        current_word = ""
        buffer.clear()
        last_letter = ""
        LETTER_LOCK = False

    # =====================================================
    # UI (UNCHANGED)
    # =====================================================

    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    cv2.putText(frame, "AI SIGN LANGUAGE TRANSLATOR",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    cv2.putText(frame, f"Letter: {predicted}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, h - 180), (w, h), (20, 20, 20), -1)

    cv2.putText(frame, f"Sentence: {sentence}",
                (20, h - 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (180, 180, 180), 2)

    cv2.putText(frame, current_word,
                (20, h - 50), cv2.FONT_HERSHEY_DUPLEX,
                2, (0, 255, 100), 3)

    cv2.imshow("AI Sign Translator", frame)

# =========================================================
# CLEANUP
# =========================================================

speech_queue.put(None)

cap.release()
cv2.destroyAllWindows()