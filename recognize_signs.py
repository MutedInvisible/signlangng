import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# my neck hurts
model = load_model("sign_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# e
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    label = "Neutral"  # no hand deault

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row = np.array(row).reshape(1, -1)
            pred = model.predict(row)
            label = le.inverse_transform([np.argmax(pred)])[0]

    cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Word Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
