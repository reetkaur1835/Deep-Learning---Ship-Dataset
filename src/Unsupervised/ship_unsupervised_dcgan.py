# ship_unsupervised_dcgan.py
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------- CONFIG --------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train", "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")

IMG_SIZE = 64   # small images for DCGAN
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS_GAN = 100   # increase for final experiments
NUM_GENERATED_PER_CLASS = 300   # number of synthetic images per class to create
SEED = 42
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "dcgan")
GENERATED_DIR = os.path.join(RESULTS_DIR, "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 1) load image paths & labels (we only use images to train GAN)
df = pd.read_csv(TRAIN_CSV)
df["image_path"] = df["image"].apply(lambda x: os.path.join(IMAGES_DIR, x))
# We'll train GAN on all images combined (not per-class) for simplicity (class-agnostic generator)
all_paths = df["image_path"].values

# Preprocess dataset for GAN: resize, scale to [-1,1]
def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img

ds = tf.data.Dataset.from_tensor_slices(all_paths)
ds = ds.map(lambda p: load_and_preprocess(p), num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.shuffle(buffer_size=len(all_paths), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 2) DCGAN model definitions (small)
def make_generator():
    model = keras.Sequential([
        layers.Dense(8*8*128, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8,8,128)),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2D(3, 3, activation='tanh', padding='same')  # output in [-1,1]
    ], name="generator")
    return model

def make_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(IMG_SIZE,IMG_SIZE,3)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ], name="discriminator")
    return model

generator = make_generator()
discriminator = make_discriminator()

# Losses and optimizers
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = keras.optimizers.Adam(1e-4)
d_optimizer = keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(real_images):
    noise = tf.random.normal([tf.shape(real_images)[0], LATENT_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        d_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                 cross_entropy(tf.zeros_like(fake_output), fake_output)
        g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return d_loss, g_loss

# Training loop (or load if already trained)
GENERATOR_PATH = os.path.join(RESULTS_DIR, "generator.h5")
DISCRIMINATOR_PATH = os.path.join(RESULTS_DIR, "discriminator.h5")

if os.path.exists(GENERATOR_PATH):
    print("Loading pre-trained GAN models...")
    generator = keras.models.load_model(GENERATOR_PATH, compile=False)
    if os.path.exists(DISCRIMINATOR_PATH):
        discriminator = keras.models.load_model(DISCRIMINATOR_PATH, compile=False)
    print("GAN models loaded successfully!")
else:
    print("Starting GAN training...")
    for epoch in range(1, EPOCHS_GAN+1):
        d_loss_avg = tf.metrics.Mean()
        g_loss_avg = tf.metrics.Mean()
        for batch in ds:
            d_loss, g_loss = train_step(batch)
            d_loss_avg.update_state(d_loss); g_loss_avg.update_state(g_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS_GAN}  d_loss={d_loss_avg.result().numpy():.4f}  g_loss={g_loss_avg.result().numpy():.4f}")
            # save sample grid
            noise = tf.random.normal([16, LATENT_DIM])
            gen_imgs = generator(noise, training=False).numpy()
            gen_imgs = (gen_imgs + 1.0) * 127.5
            fig, axes = plt.subplots(4,4, figsize=(6,6))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(np.clip(gen_imgs[i].astype(np.uint8),0,255))
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"sample_epoch_{epoch}.png"), dpi=150)
            plt.close()

    # Save generator and discriminator
    generator.save(GENERATOR_PATH)
    discriminator.save(DISCRIMINATOR_PATH)
    print("GAN training finished, models saved.")

# 3) Generate synthetic images and save into class subfolders by copying real labels
# We'll generate images and then assign them to classes by sampling existing labels proportionally.
print("Generating synthetic images for augmentation...")
label_counts = df["category"].value_counts().to_dict()
# create subfolders for generated images named by class index (1..5)
for cls in sorted(df["category"].unique()):
    os.makedirs(os.path.join(GENERATED_DIR, str(cls)), exist_ok=True)

# generate NUM_GENERATED_PER_CLASS * num_classes images total
num_classes = len(label_counts)
total_to_gen = NUM_GENERATED_PER_CLASS * num_classes
batch_size_gen = 64
generated = 0
while generated < total_to_gen:
    n = min(batch_size_gen, total_to_gen - generated)
    noise = tf.random.normal([n, LATENT_DIM])
    imgs = generator(noise, training=False).numpy()
    imgs = (imgs + 1.0) * 127.5
    for i in range(n):
        cls = random.choice(list(label_counts.keys()))
        out_path = os.path.join(GENERATED_DIR, str(cls), f"gen_{generated+i:06d}.png")
        plt.imsave(out_path, np.clip(imgs[i].astype(np.uint8),0,255))
    generated += n
print(f"Generated {generated} synthetic images saved to {GENERATED_DIR}")

# 4) Retrain a classifier on augmented data (original + generated)
# Build a dataset that reads both real and generated images and their labels
def gather_augmented_records():
    records = []
    # real
    for _, row in df.iterrows():
        records.append((row["image_path"], int(row["category"])-1))
    # generated
    for cls in sorted(df["category"].unique()):
        gen_folder = os.path.join(GENERATED_DIR, str(cls))
        for p in glob(os.path.join(gen_folder, "*.png")):
            records.append((p, int(cls)-1))
    rec_df = pd.DataFrame(records, columns=["path","label"])
    return rec_df

aug_df = gather_augmented_records()
train_df2, val_df2 = train_test_split(aug_df, test_size=0.2, random_state=SEED, stratify=aug_df["label"])
train_paths = train_df2["path"].values; train_labels = train_df2["label"].values
val_paths = val_df2["path"].values; val_labels = val_df2["label"].values

# pipeline (resize back to 128 for classifier)
IMG_CLS = 128
def decode_img_cls(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_CLS, IMG_CLS])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def make_ds_cls(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_img_cls, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds_aug = make_ds_cls(train_paths, train_labels, shuffle=True)
val_ds_aug = make_ds_cls(val_paths, val_labels, shuffle=False)

# small CNN classifier (similar to your baseline but smaller)
def build_small_cnn(input_shape=(IMG_CLS,IMG_CLS,3), num_classes=num_classes):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32,3,activation="relu",padding="same")(inp); x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,3,activation="relu",padding="same")(x); x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128,3,activation="relu",padding="same")(x); x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="small_cnn_aug")

clf = build_small_cnn()
clf.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = clf.fit(train_ds_aug, validation_data=val_ds_aug, epochs=12)

# evaluate
y_true = np.concatenate([y.numpy() for _, y in val_ds_aug], axis=0)
y_probs = clf.predict(val_ds_aug)
y_pred = np.argmax(y_probs, axis=1)
acc = np.mean(y_pred == y_true)
print(f"Augmented classifier validation accuracy: {acc:.4f}")

report = classification_report(y_true, y_pred, digits=4)
with open(os.path.join(RESULTS_DIR, "classification_report_augmented.txt"), "w") as f:
    f.write(report)
print(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues"); plt.title("Confusion Matrix - Augmented"); plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_augmented.png"), dpi=150); plt.show()

print("DCGAN augmentation + classifier results saved to:", RESULTS_DIR)
