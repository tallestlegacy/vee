# imports
import fastapi
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np


# gobal variable
model_path = "./eczema_1.h5"

# tensorflow details
print("Using tensorflow version " + tf.__version__)

# load model
print("Loading model")
loaded_model = load_model(model_path)
loaded_model.summary()


# Helper function


# image pre-processor
def preprocess_image(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# image loader
def load_img_from_path(path):
    img = image.load_img(path, target_size=(224, 224))
    return preprocess_image(img)


def make_prediction(image):
    preds = loaded_model.predict(image)
    return preds


def classify(image):
    predictions = loaded_model.predict(image)

    class_names = [
        "Atopic Dermatitis",
        "Prioasis",
    ]  # Replace with your actual class class_names
    predicted_class_index = int(predictions[0][0] > 0.5)  # Assuming threshold is 0.5
    predicted_class_label = class_names[predicted_class_index]
    confidence = (
        predictions[0][0] if predicted_class_index == 1 else 1 - predictions[0][0]
    )

    return {"predicted_class": predicted_class_label, "confidence": confidence}


# test
print(classify(load_img_from_path("./test/test_p.jpg")))


# API
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def prediction():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    print("File received: " + file.filename)

    if file:
        img = Image.open(file)
        print("Making prediction")
        processed_image = preprocess_image(img)
        res = classify(processed_image)
        print(f"Should return {res}")
        return jsonify(
            {
                "predicted_class": res["predicted_class"],
                "confidence": "{:.2f}%".format(res["confidence"] * 100),
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
