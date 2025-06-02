import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model and classes
model = load_model('model/asl_model.h5')
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Initialize MediaPipe and Webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            x_min = max(0, int(x_min) - 20)
            y_min = max(0, int(y_min) - 20)
            x_max = min(w, int(x_max) + 20)
            y_max = min(h, int(y_max) + 20)

            hand_img = frame[y_min:y_max, x_min:x_max]

            try:
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = np.expand_dims(hand_img, axis=0)
                hand_img = hand_img / 255.0

                predictions = model.predict(hand_img)
                predicted_class = np.argmax(predictions)
                predicted_letter = classes[predicted_class]

                cv2.putText(frame, f'Letter: {predicted_letter}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            except:
                pass

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Hand Sign Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
