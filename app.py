from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os
from PIL import Image
import pandas as pd
from flask_cors import CORS
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app, origins=["*"])

model = tf.keras.models.load_model(
    "./models/pTrainModel10.keras", custom_objects={"KerasLayer": hub.KerasLayer}
)

model1 = tf.keras.models.load_model(
    "./models/xrayOrNot69.keras", custom_objects={"KerasLayer": hub.KerasLayer}
)

# Assuming your CSV file has a column named "Label" containing the labels
# Load the CSV file
labels_df = pd.read_csv("tempLabels.csv")

# Extract the labels column
test_labels = labels_df["target"].tolist()


@app.route("/")
def home():
    return "successful Flask app."


@app.route("/predict", methods=["POST"])
def predict():
    # Get the uploaded image file
    image_file = request.files["image"]
    print(image_file)

    if image_file.filename == "":
        return "No file selected"

    # Save the file to a temporary location
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(upload_path)
    image_array = Image.open(os.path.join("./uploads", image_file.filename))
    print("image", image_array)

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(
        os.path.join("./uploads", image_file.filename), target_size=(224, 224)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Make predictions with model1
    predictions_model1 = model1.predict(image)
    probability_model1 = predictions_model1[0][0]
    if probability_model1 > 0.7:
        result_model1 = False
    else:
        result_model1 = True

    # Calculate accuracy for model1
    # model1_pred_labels = int(result_model1)
    # model1_accuracy = accuracy_score(test_labels, [model1_pred_labels])

    return jsonify(
        {
            "status": True,
            "data": result_model1,
        }
    )


@app.route("/detect", methods=["POST"])
def detect():
    # Get the uploaded image file
    image_file = request.files["image"]
    print(image_file)

    if image_file.filename == "":
        return "No file selected"

    # Save the file to a temporary location
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(upload_path)
    image_array = Image.open(os.path.join("./uploads", image_file.filename))
    print("image", image_array)

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(
        os.path.join("./uploads", image_file.filename), target_size=(224, 224)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    print(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Make predictions with model
    predictions_model = model.predict(image)
    probability_model = predictions_model[0][0]
    if probability_model > 0.8:
        result_model = "No Pneumonia Detected"
    else:
        result_model = "Pneumonia Detected"

    # Calculate accuracy for model
    # model_pred_labels = int(probability_model <= 0.8)
    # model_accuracy = accuracy_score(test_labels, [model_pred_labels])

    return jsonify(
        {
            "status": True,
            "data": result_model,
            "accuracy": str(1 - predictions_model[0][0]),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
