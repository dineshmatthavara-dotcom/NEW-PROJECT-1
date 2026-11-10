
import sqlite3
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI(title="EcoVision AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = tf.keras.models.load_model("model/waste_model.h5")

# Load class labels
with open("model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

@app.get("/")
def home():
    return {"message": "EcoVision AI is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert("RGB")

        height, width = model.input_shape[1:3]
        image = image.resize((width, height))

        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)

        index = int(np.argmax(prediction))
        category = class_names[index]
        confidence = float(prediction[0][index]) * 100

        return {
            "category": category,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}
# Database setup
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "predictions.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def create_database():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

create_database()