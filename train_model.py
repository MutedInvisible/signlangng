import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pickle


df = pd.read_csv("sign_data_clean.csv", header=None)


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ts hard
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# me
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# smart stuff
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
print("Training model...")
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save("sign_model.h5")
print("Model trained and saved as sign_model.h5")
