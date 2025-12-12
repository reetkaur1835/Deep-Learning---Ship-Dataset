import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Set project root (go up three levels from current file location)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Model parameters
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20

# Data paths
CSV_PATH = os.path.join(DATA_DIR, 'train', 'train.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'images')

# Create results directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'unsupervised_encoder')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- DATA LOADING (No Labels Needed!) ---
def load_data_unsupervised(csv_path, img_dir, img_size):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        print(f"Current working directory: {os.getcwd()}")
        raise
        
    images = []
    print(f"Found {len(df)} samples. Loading images from {img_dir}...")
    
    for index, row in df.iterrows():
        try:
            img_path = os.path.join(img_dir, row['image'])
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping...")
                continue
                
            img = load_img(img_path, target_size=(img_size, img_size))
            img_array = img_to_array(img)
            images.append(img_array)
            
            # Print progress
            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1}/{len(df)} images...")
                
        except Exception as e:
            print(f"Error processing {row.get('image', 'unknown')}: {str(e)}")
            continue
            
    if not images:
        raise ValueError("No images were successfully loaded. Please check your file paths.")
        
    print(f"Successfully loaded {len(images)} images.")
    return np.array(images)

X = load_data_unsupervised(CSV_PATH, IMAGES_DIR, IMG_SIZE)
X = X / 255.0 # Normalize
X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

# --- BUILD AUTOENCODER ---
def build_autoencoder():
    input_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # --- ENCODER (The Feature Extractor) ---
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # 64x64
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # 32x32
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same', name='encoder_output')(x) # 16x16
    
    # --- DECODER (The Reconstructor) ---
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x) # 32x32
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 64x64
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 128x128
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Full Autoencoder
    autoencoder = models.Model(input_img, decoded)
    
    # Separate Encoder Model (For Transfer Learning later)
    encoder = models.Model(input_img, encoded)
    
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error is best for reconstruction
autoencoder.summary()

# --- TRAIN UNSUPERVISED ---
# Note: y is X_train because target is the image itself!
history = autoencoder.fit(X_train, X_train,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          validation_data=(X_val, X_val))

# --- SAVE EVERYTHING ---
os.makedirs('results', exist_ok=True)
autoencoder.save('results/ship_autoencoder_full.keras')
encoder.save('results/ship_encoder_only.keras') # <--- THIS IS THE GOLD FOR STEP 2
print("Encoder saved for Transfer Learning!")

# --- VISUALIZE RECONSTRUCTION ---
decoded_imgs = autoencoder.predict(X_val[:5])
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_val[i])
    plt.axis("off")
    # Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.axis("off")
plt.savefig('results/ae_reconstruction_sample.png')
