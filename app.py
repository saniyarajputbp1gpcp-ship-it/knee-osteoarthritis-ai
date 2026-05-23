from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

# Load trained model
model = load_model("improved_model.keras", compile=False)

# Upload folder
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
classes = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

IMG_SIZE = 224

# Home page
@app.route('/')
def home():

    return render_template("index.html")


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Check if file exists
    if 'file' not in request.files:

        return "No file uploaded"

    # Get uploaded file
    file = request.files['file']

    # Patient details
    name = request.form['name']

    age = request.form['age']

    gender = request.form['gender']

    height = float(request.form['height'])

    weight = float(request.form['weight'])

    symptoms = request.form['symptoms']

    # BMI calculation
    bmi = round(weight / (height * height), 2)

    # Check filename
    if file.filename == '':

        return "No selected file"

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    # Read image
    img = cv2.imread(filepath)

    # Resize image
    img = cv2.resize(img, (224, 224))

    # Normalize image
    img = img / 255.0

    # Reshape image
    img = np.reshape(img, (1, 224, 224, 3))

    # Predict
    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result = classes[predicted_class]

    # Render result page
    return render_template(

        "result.html",

        prediction=result,

        confidence=round(confidence, 2),

        image_path=filepath,

        name=name,

        age=age,

        gender=gender,

        bmi=bmi,

        symptoms=symptoms
    )


# Run app
if __name__ == '__main__':

    app.run(debug=True)