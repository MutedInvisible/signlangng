import cv2
import mediapipe as mp
import pandas as pd
from tkinter import simpledialog, Tk


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

root = Tk()
root.withdraw() 

data = []

cap = cv2.VideoCapture(0)

while True:
    # ts OMG
    label = simpledialog.askstring("Label", "Enter label name for this gesture (or 'quit' to stop):")
    if label is None or label.lower() == "quit":
        break

    print(f"Recording {label}... Move your hand in front of the camera.")

    # Record 50 frames per action u do
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        row = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
            row.append(label)  
            data.append(row)

        cv2.putText(frame, f"{label} ({i+1}/50)", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("sign_data.csv", index=False, header=False)
print("Recording complete! Data saved to sign_data.csv")
