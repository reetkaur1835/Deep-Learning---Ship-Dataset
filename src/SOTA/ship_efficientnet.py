import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMG_SIZE = 128 # EfficientNet is flexible, B0 works well with 128-224
BATCH_SIZE = 32
EPOCHS = 15

# Get project root and set correct paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "train", "images")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "train", "train.csv")

# --- DATA LOADING ---
def load_data(csv_path, img_dir, img_size):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    print("Loading images for EfficientNet...")
    for index, row in df.iterrows():
        try:
            img = load_img(os.path.join(img_dir, row['image']), 
                           target_size=(img_size, img_size))
            img_array = img_to_array(img)
            # EfficientNet expects 0-255 inputs (it handles scaling internally usually, 
            # but standardizing to 0-255 input is safe)
            images.append(img_array)
            labels.append(row['category'] - 1)
        except:
            pass
    return np.array(images), np.array(labels)

X, y = load_data(CSV_PATH, DATA_DIR, IMG_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BUILD MODEL ---
def build_efficientnet_model():
    # 1. Load Base Model
    # include_top=False removes the 1000-class ImageNet layer
    base_model = EfficientNetB0(weights='imagenet', 
                                include_top=False, 
                                input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # 2. Freeze Base
    base_model.trainable = False 
    
    # 3. Add New Head
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='softmax'))
    
    # 4. Compile
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_efficientnet_model()
model.summary()

# --- TRAIN ---
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val))

# --- SAVE ---
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
model.save(os.path.join(RESULTS_DIR, 'ship_efficientnet.keras'))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('EfficientNetB0 Transfer Learning Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'efficientnet_curves.png'))
print("EfficientNet Model Saved.")