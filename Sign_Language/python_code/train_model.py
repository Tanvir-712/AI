import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ============================================
# CREATE MODEL DIRECTORY
# ============================================

os.makedirs("model", exist_ok=True)

# ============================================
# LOAD DATASET
# ============================================

print("\nLoading dataset...")

df = pd.read_csv("dataset.csv", header=None)

# Features
X = df.iloc[:, :-1].values.astype("float32")

# Labels
y = df.iloc[:, -1].values

print(f"Dataset Loaded: {len(df)} samples")

# ============================================
# ENCODE LABELS
# ============================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# Save labels
np.save("model/labels.npy", encoder.classes_)

print("Labels Saved")

# One-hot encoding
y_categorical = to_categorical(y_encoded)

# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# ============================================
# BUILD MODEL
# ============================================

print("\nBuilding model...")

model = Sequential([

    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.3),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(len(encoder.classes_), activation='softmax')
])

# ============================================
# COMPILE MODEL
# ============================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# SAVE BEST MODEL
# ============================================

checkpoint = ModelCheckpoint(
    filepath="model/sign_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ============================================
# TRAIN MODEL
# ============================================

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint],
    verbose=1
)

# ============================================
# EVALUATE MODEL
# ============================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

# ============================================
# SAVE FINAL MODEL
# ============================================

model.save("model/final_model.h5")

print("\nTraining Complete!")
print("Saved:")
print(" - model/sign_model.h5")
print(" - model/final_model.h5")
print(" - model/labels.npy")