import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from recommendation import cnv, dme, drusen, normal
from PIL import Image
import io

app = FastAPI(title="OCT Eye Disease Predictor")

# Rebuild model architecture (matching Training_model.ipynb) and load weights
# This avoids Keras version deserialization issues with the saved model
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=True,
    include_preprocessing=True,
    classes=1000,
)
model = Sequential([base_model, Dense(units=4, activation="softmax")])
model.load_weights("Trained_Model.h5")

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
RECOMMENDATIONS = [cnv, dme, drusen, normal]


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)[0]
    result_index = int(np.argmax(preds))
    confidence_all = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(4)}

    return {
        "prediction": CLASS_NAMES[result_index],
        "confidence": confidence_all,
        "recommendation": RECOMMENDATIONS[result_index],
    }


app.mount("/static", StaticFiles(directory="static"), name="static")
