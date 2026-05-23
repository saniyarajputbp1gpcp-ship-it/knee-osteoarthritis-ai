import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("improved_model.h5")

# Class labels
classes = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# Image size
IMG_SIZE = 224

# Image path
image_path = "test_image.jpg"

# Read image
img = cv2.imread(image_path)

# Resize image
img = cv2.resize(img, (224,224))

# Normalize
img = img / 255.0

# Reshape for CNN
img = np.reshape(img, (1,224,224,3))

# Predict
prediction = model.predict(img)

# Get class index
predicted_class = np.argmax(prediction)

# Confidence
confidence = np.max(prediction) * 100

# Result
print("\nPrediction Result")
print("------------------")

print("Class:", predicted_class)
print("Severity:", classes[predicted_class])

print(f"Confidence: {confidence:.2f}%")