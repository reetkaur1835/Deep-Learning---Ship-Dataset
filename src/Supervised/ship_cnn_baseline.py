import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
layers = keras.layers

# CONFIG
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# CSV is here:
TRAIN_CSV = os.path.join(DATA_DIR, "train", "train.csv")

# IMAGES ARE HERE:
IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")


IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 5   # 5 ship classes
EPOCHS = 5        # you can increase later

# Save results in the root directory of the project
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set all random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'

# Configure TensorFlow for deterministic operations
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
tf.keras.utils.set_random_seed(SEED)
# tf.config.experimental.enable_op_determinism()

print(f"TensorFlow Version: {tf.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU devices found: {gpus}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU devices found - training will use CPU")

# -------------------------------------------------------------------
# 1. LOAD CSV & BUILD PATHS
# -------------------------------------------------------------------
print(">>> Loading train.csv...")
df = pd.read_csv(TRAIN_CSV)
print("DataFrame head:")
print(df.head())

#  CREATE image_path COLUMN *BEFORE* USING IT
df["image_path"] = df["image"].apply(lambda x: os.path.join(IMAGES_DIR, x))

print("\n>>> Checking image paths for first few rows...")
for p in df["image_path"].head(5):
    print(p, "->", os.path.exists(p))


# -------------------------------------------------------------------
# 2. TRAIN / VALIDATION SPLIT
# -------------------------------------------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["category"]
)

print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}")

train_paths = train_df["image_path"].values
train_labels = train_df["category"].values

val_paths = val_df["image_path"].values
val_labels = val_df["category"].values

# -------------------------------------------------------------------
# 3. TF.DATA PIPELINE
# -------------------------------------------------------------------
def preprocess_label(label):
    # Original labels assumed 1..5 -> convert to 0..4
    return label - 1

def decode_img(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def process_example(path, label):
    img = decode_img(path)
    label = preprocess_label(label)
    return img, label

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(process_example, num_parallel_calls=4)
    # Cache images in RAM to speed up subsequent epochs
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

print("\n>>> Building tf.data datasets...")
train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels, shuffle=False)

for batch_imgs, batch_labels in train_ds.take(1):
    print("Example batch images shape:", batch_imgs.shape)
    print("Example batch labels shape:", batch_labels.shape)

# -------------------------------------------------------------------
# 4. SHOW SAMPLE IMAGES
# -------------------------------------------------------------------
print("\n>>> Displaying sample images...")
sample_batch = next(iter(train_ds))
sample_imgs, sample_lbls = sample_batch

# Map indices (0..4) to class names
idx_to_name = {
    0: "Cargo",
    1: "Military",
    2: "Carrier",
    3: "Cruise",
    4: "Tankers"
}

plt.figure(figsize=(10, 10))
num_to_show = min(9, sample_imgs.shape[0])
for i in range(num_to_show):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(sample_imgs[i])
    label_idx = int(sample_lbls[i].numpy())
    plt.title(idx_to_name.get(label_idx, str(label_idx)))
    plt.axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 5. SIMPLE CNN MODEL
# -------------------------------------------------------------------
def build_simple_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="simple_ship_cnn")
    return model

print("\n>>> Building model...")
model = build_simple_cnn()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# -------------------------------------------------------------------
# 6. TRAIN THE MODEL (or load if already trained)
# -------------------------------------------------------------------
MODEL_PATH = os.path.join(RESULTS_DIR, "baseline_cnn.h5")

if os.path.exists(MODEL_PATH):
    print("\n>>> Loading pre-trained model...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("Model loaded successfully!")
    # Evaluate to get metrics
    print("\n>>> Evaluating loaded model...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation accuracy: {val_acc:.4f}")
    history = None
else:
    print("\n>>> Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1  # Show progress bar with metrics (0 = silent, 1 = progress bar, 2 = one line per epoch)
    )
    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

# -------------------------------------------------------------------
# 7. PLOT ACCURACY & LOSS (only if we just trained)
# -------------------------------------------------------------------
if history is not None:
    print("\n>>> Plotting training curves...")

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    # Use the OO API 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy subplot
    ax1.plot(epochs_range, acc, label="Train Acc")
    ax1.plot(epochs_range, val_acc, label="Val Acc")
    ax1.set_title("Training & Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss subplot
    ax2.plot(epochs_range, loss, label="Train Loss")
    ax2.plot(epochs_range, val_loss, label="Val Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.tight_layout()

    # Save the *same* figure we just drew on
    plot_path = os.path.join(RESULTS_DIR, "training_curves.png")
    fig.savefig(plot_path, dpi=150)

    # And also show it on screen
    plt.show()

    print(f"\nTraining curves saved to: {plot_path}")
else:
    print("\n>>> Skipping training curves (loaded pre-trained model)")

print("\nDone!")
