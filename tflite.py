import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import flask_cors

# Global variables
model_path = "./model.tflite"  # Change the path to your TFLite model file

# TensorFlow details
print("Using tensorflow version " + tf.__version__)

# Load TFLite model
print("Loading TFLite model")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Helper functions


# Image pre-processor
def preprocess_image(img):
    img = img.resize(
        (224, 224)
    )  # Resize image to match the input size expected by the model
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def classify(image):
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    interpreter.set_tensor(input_tensor_index, image)
    interpreter.invoke()
    predictions = output()[0]

    class_names = [
        "Atopic Dermatitis",
        "Psoriasis",
    ]  # Replace with your actual class names
    predicted_class_index = int(predictions > 0.5)  # Assuming threshold is 0.5
    predicted_class_label = class_names[predicted_class_index]
    confidence = predictions[0] if predicted_class_index == 1 else 1 - predictions[0]

    return {"predicted_class": predicted_class_label, "confidence": confidence}


# API
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
flask_cors.CORS(app)


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


app.run(debug=True, port=8080)
