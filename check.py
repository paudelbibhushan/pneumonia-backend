import os
import tensorflow as tf
import numpy as np

# Set the path to the folder containing the images
folder_path = "C:\\Users\\user\\Downloads\\X_RAY\\chest_xray\\test\\NORMAL"


# Load the trained model
model = tf.keras.models.load_model("models/pTrainedModel.keras")

# Create a list to store the image file names and their corresponding predictions
predictions = []

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    # Set the path to the current file
    file_path = os.path.join(folder_path, file_name)

    # Preprocess the image file as needed (e.g., resize, normalize, etc.)
    # ...

    # Load and preprocess the image file
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    # Make a prediction using the loaded model
    prediction = model.predict(img)

    # Get the predicted class label
    if prediction > 0.5:
        label = "Pneumonia"
    else:
        label = "Normal"

    # Append the file name and prediction to the list
    predictions.append((file_name, label))

# Print the predictions
for file_name, label in predictions:
    print(f"File: {file_name}, Prediction: {label}")
