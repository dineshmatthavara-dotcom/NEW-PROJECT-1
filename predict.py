import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model(
    "model/waste_model.h5"
)

# Load class labels
with open("model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

# Select image
image_path = "test_image.jpg"

# Get expected image size
height, width = model.input_shape[1:3]

# Load image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(height, width)
)

# Convert image
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

index = np.argmax(prediction)

print("Predicted category:", class_names[index])
print("Confidence:", round(float(prediction[0][index]) * 100, 2), "%")
index = int(np.argmax(prediction))
category = class_names[index]
confidence = float(prediction[0][index]) * 100
# Store actual AI prediction
conn = sqlite3.connect(DB_PATH)

conn.execute(
    """
    INSERT INTO predictions (category, confidence, created_at)
    VALUES (?, ?, ?)
    """,
    (
        category,
        confidence,
        datetime.now().isoformat(timespec="seconds")
    )
)

conn.commit()
conn.close()