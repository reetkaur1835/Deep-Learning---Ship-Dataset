import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = '/Users/reet/Desktop/cent courses/Fall2025/Deep Learning/Deep-Learning---Ship-Dataset/data/train/images' 
CSV_PATH = '/Users/reet/Desktop/cent courses/Fall2025/Deep Learning/Deep-Learning---Ship-Dataset/data/train/train.csv'

# --- DATA LOADING (With Labels this time!) ---
def load_data(csv_path, img_dir, img_size):
    print(f"Loading data from CSV: {csv_path}")
    print(f"Looking for images in: {img_dir}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV")
    
    # Display first few rows of the CSV
    print("\nFirst few rows of the CSV:")
    print(df.head())
    
    images = []
    labels = []
    failed_loads = 0
    
    print("\nStarting to load images...")
    for index, row in df.iterrows():
        try:
            img_path = os.path.join(img_dir, row['image'])
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                failed_loads += 1
                continue
                
            img = load_img(img_path, target_size=(img_size, img_size))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(row['category'] - 1)
            
            # Print progress
            if (len(images) % 100) == 0:
                print(f"Loaded {len(images)} images...")
                
        except Exception as e:
            print(f"Error loading {row['image']}: {str(e)}")
            failed_loads += 1
    
    print(f"\nSuccessfully loaded {len(images)} images")
    print(f"Failed to load {failed_loads} images")
    
    if len(images) == 0:
        print("\nWARNING: No images were loaded!")
        print("Please check the following:")
        print(f"1. CSV file exists at: {csv_path}")
        print(f"2. Image directory exists: {img_dir}")
        print(f"3. The 'image' column in the CSV contains correct relative paths to the images")
        print(f"4. The images exist at the specified paths")
    return np.array(images), np.array(labels)

X, y = load_data(CSV_PATH, DATA_DIR, IMG_SIZE)
X = X / 255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LOAD THE PRE-TRAINED ENCODER ---
print("Loading pre-trained encoder...")
try:
    pretrained_encoder = models.load_model('results/ship_encoder_only.keras')
    pretrained_encoder.trainable = False # FREEZE the weights!
    print("Encoder loaded and frozen.")
except:
    print("ERROR: Run ship_unsupervised_ae.py first to generate the encoder!")
    exit()

# --- BUILD CLASSIFIER ON TOP ---
model = models.Sequential()
model.add(pretrained_encoder) # The "Transfer" happens here
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- TRAIN (Only the new Dense layers will train) ---
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    verbose=2)
                

# --- SAVE & PLOT ---
model.save('results/ship_transfer_model.keras')

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Unsupervised Transfer Learning Performance')
plt.legend()
plt.savefig('results/transfer_learning_curves.png')
print("Done! You have successfully used Unsupervised Transfer Learning.")