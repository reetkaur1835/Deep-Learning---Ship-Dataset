import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMG_SIZE = 224  # ResNet was designed for 224x224
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
    print("Loading images for ResNet50...")
    for index, row in df.iterrows():
        try:
            img = load_img(os.path.join(img_dir, row['image']), 
                           target_size=(img_size, img_size))
            img_array = img_to_array(img)
            # ResNet specific preprocessing (very important!)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            images.append(img_array)
            labels.append(row['category'] - 1)
        except:
            pass
    return np.array(images), np.array(labels)

X, y = load_data(CSV_PATH, DATA_DIR, IMG_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BUILD MODEL ---
def build_resnet50_model():
    # 1. Load the Base Model (Pre-trained on ImageNet)
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, # Drop the final classification layer
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # 2. Freeze the Base (Keep the pre-learned patterns)
    base_model.trainable = False 
    
    # 3. Add New Head (For your 5 Ship Classes)
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D()) # Condense features
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))             # Regularization
    model.add(layers.Dense(5, activation='softmax')) # Output
    
    # 4. Compile
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_resnet50_model()
model.summary()

# --- TRAIN ---
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val))

# --- SAVE ---
os.makedirs('results', exist_ok=True)
model.save('results/ship_resnet50.keras')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('ResNet50 Transfer Learning Accuracy')
plt.legend()
plt.savefig('results/resnet50_curves.png')
print("ResNet50 Model Saved.")