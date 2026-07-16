import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import TRAIN_DATASET_PATH, IMG_SIZE

# Dataset Path
DATASET_PATH = Path(TRAIN_DATASET_PATH)

# Load image paths
glioma = list((DATASET_PATH / "glioma").glob("*"))
meningioma = list((DATASET_PATH / "meningioma").glob("*"))
notumor = list((DATASET_PATH / "notumor").glob("*"))
pituitary = list((DATASET_PATH / "pituitary").glob("*"))

random.seed(42)

print("Glioma:     ", len(glioma))
print("Meningioma: ", len(meningioma))
print("No Tumor:   ", len(notumor))
print("Pituitary:  ", len(pituitary))

# Randomly sample 1400 images from each class
glioma = random.sample(glioma, 1400)
meningioma = random.sample(meningioma, 1400)
notumor = random.sample(notumor, 1400)
pituitary = random.sample(pituitary, 1400)

print("Glioma:     ", len(glioma))
print("Meningioma: ", len(meningioma))
print("No Tumor:   ", len(notumor))
print("Pituitary:  ", len(pituitary))

# Create file and label lists
files = []
labels = []

for f in glioma:
    files.append(str(f))
    labels.append(0)

for f in meningioma:
    files.append(str(f))
    labels.append(1)

for f in notumor:
    files.append(str(f))
    labels.append(2)

for f in pituitary:
    files.append(str(f))
    labels.append(3)

print("Total images:", len(files))

# Train / Validation / Test split
train_files, test_files, train_labels, test_labels = train_test_split(
    files,
    labels,
    test_size=0.15,
    random_state=42,
    stratify=labels
)

train_files, val_files, train_labels, val_labels = train_test_split(
    train_files,
    train_labels,
    test_size=0.15,
    random_state=42,
    stratify=train_labels
)

print(len(train_files), len(val_files), len(test_files))


# Image preprocessing
def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


# Create TensorFlow datasets
def create_dataset(files, labels):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = create_dataset(train_files, train_labels)
val_ds = create_dataset(val_files, val_labels)
test_ds = create_dataset(test_files, test_labels)

# Visualize sample images
label_names = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

plt.figure(figsize=(8, 8))

for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(label_names[labels[i].numpy()])
        plt.axis("off")

plt.show()