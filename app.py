import flask_cors
from flask import Flask, render_template, request, jsonify
from PIL import Image

from utilities import init_model

(preprocess_image, classify) = init_model()

app = Flask(__name__)
flask_cors.CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def prediction():
    try:
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
    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify({"error": "Something went wrong"})


if __name__ == "__main__":
    app.run(debug=True)
