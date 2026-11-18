
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------
# 1. PROJECT PATHS
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "train"
VALIDATION_DIR = BASE_DIR / "validation"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------
# 2. TRAINING SETTINGS
# ---------------------------------

IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 10

# ---------------------------------
# 3. CHECK FOLDERS
# ---------------------------------

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training folder missing: {TRAIN_DIR}")

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder missing: {VALIDATION_DIR}"
    )

train_classes = sorted(
    p.name for p in TRAIN_DIR.iterdir() if p.is_dir()
)

validation_classes = sorted(
    p.name for p in VALIDATION_DIR.iterdir() if p.is_dir()
)

if not train_classes:
    raise ValueError("No category folders found inside train!")

if train_classes != validation_classes:
    raise ValueError(
        f"Category mismatch!\n"
        f"Train: {train_classes}\n"
        f"Validation: {validation_classes}"
    )

print("Waste categories:", train_classes)

# ---------------------------------
# 4. LOAD TRAINING IMAGES
# ---------------------------------

train_data = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="int",
    class_names=train_classes,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

# ---------------------------------
# 5. LOAD VALIDATION IMAGES
# ---------------------------------

validation_data = tf.keras.utils.image_dataset_from_directory(
    str(VALIDATION_DIR),
    labels="inferred",
    label_mode="int",
    class_names=train_classes,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------------------------
# 6. IMPROVE DATA LOADING
# ---------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.prefetch(
    buffer_size=AUTOTUNE
)

# ---------------------------------
# 7. BUILD CNN MODEL
# ---------------------------------

model = models.Sequential([
    layers.Input(shape=(180, 180, 3)),

    layers.Rescaling(1./255),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(
        len(train_classes),
        activation="softmax"
    )
])

# ---------------------------------
# 8. COMPILE MODEL
# ---------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------
# 9. TRAIN MODEL
# ---------------------------------

print("\nStarting EcoVision-AI training...")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# ---------------------------------
# 10. SAVE TRAINED MODEL
# ---------------------------------

model.save(str(MODEL_DIR / "waste_model.h5"))

# ---------------------------------
# 11. SAVE CATEGORY LABELS
# ---------------------------------

with open(MODEL_DIR / "labels.txt", "w") as file:
    file.write("\n".join(train_classes))

print("\nTraining completed successfully!")
print("Model saved:", MODEL_DIR / "waste_model.h5")
print("Labels saved:", MODEL_DIR / "labels.txt")

# ---------------------------------
# 12. FINAL VALIDATION
# ---------------------------------

loss, accuracy = model.evaluate(validation_data)

print(f"Validation accuracy: {accuracy * 100:.2f}%")
import os

os.makedirs("model", exist_ok=True)

model.save("model/waste_model.h5")

print("Model saved successfully!")