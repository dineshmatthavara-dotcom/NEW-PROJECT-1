import tensorflow as tf
import numpy as np
import os

if not os.path.exists("model/waste_model.h5"):
    print("Model file not found!")
    exit()

model = tf.keras.models.load_model("model/waste_model.h5")

print("Model loaded successfully!")
model.summary()