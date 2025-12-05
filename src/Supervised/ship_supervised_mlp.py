import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, utils, regularizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import random

# --- 1. CONFIGURATION ---
# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
utils.set_random_seed(SEED)

# Configure TensorFlow for deterministic operations
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
IMG_SIZE = 64  # Smaller size for MLP
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_CSV = os.path.join(DATA_DIR, "train", "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")

# --- 2. DATA LOADING ---
def load_data(csv_path, img_dir, img_size):
    """Load and preprocess the dataset."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    print(f"Found {len(df)} samples. Loading images...")
    for index, row in df.iterrows():
        try:
            img_path = os.path.join(img_dir, row['image'])
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping...")
                continue
                
            # Load and preprocess image
            img = load_img(img_path, target_size=(img_size, img_size))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(row['category'] - 1)  # Convert to 0-based indexing
            
            # Print progress
            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1}/{len(df)} images...")
                
        except Exception as e:
            print(f"Error processing {row.get('image', 'unknown')}: {str(e)}")
            continue
            
    return np.array(images), np.array(labels)

# Load and preprocess data
print("\n>>> Loading dataset...")
X, y = load_data(TRAIN_CSV, IMAGES_DIR, IMG_SIZE)
X = X / 255.0  # Normalize pixel values

# Split into training and validation sets
print("\n>>> Splitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=SEED, 
    stratify=y
)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# --- 3. DEEP MLP ARCHITECTURE ---
def create_deep_mlp(input_shape=(64, 64, 3), num_classes=5):
    """Create a deep MLP model for ship classification."""
    model = models.Sequential(name="ship_mlp")
    
    # Input layer
    model.add(layers.Flatten(input_shape=input_shape))
    
    # Hidden layers with batch normalization and dropout
    model.add(layers.Dense(2048, activation='relu', 
                          kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Dense(1024, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and compile model
print("\n>>> Creating model...")
model = create_deep_mlp(input_shape=(IMG_SIZE, IMG_SIZE, 3))
model.summary()

# --- 4. TRAINING ---
print("\n>>> Starting training...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=2  # One line per epoch
)

# --- 5. SAVE MODEL ---
model_path = os.path.join(RESULTS_DIR, "ship_mlp_model.keras")
model.save(model_path)
print(f"\n>>> Model saved to {model_path}")

# --- 6. VISUALIZATION ---
print("\n>>> Generating training curves...")
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Save figure
plot_path = os.path.join(RESULTS_DIR, "mlp_training_curves.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n>>> Training curves saved to {plot_path}")
print("\n>>> Training completed successfully!")