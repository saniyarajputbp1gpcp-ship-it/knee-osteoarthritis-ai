import os
import cv2
import numpy as np

# Dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

# Image size
IMG_SIZE = 224

# Lists to store data
X_train = []
y_train = []

X_test = []
y_test = []

# Function to load images
def load_data(folder_path, X, y):

    for label in os.listdir(folder_path):

        label_path = os.path.join(folder_path, label)

        if os.path.isdir(label_path):

            for img_name in os.listdir(label_path):

                img_path = os.path.join(label_path, img_name)

                try:
                    # Read image
                    img = cv2.imread(img_path)

                    # Resize image
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                    # Normalize image
                    img = img / 255.0

                    # Store image and label
                    X.append(img)
                    y.append(int(label))

                except Exception as e:
                    print(f"Error loading {img_path}")

# Load training data
print("Loading training data...")
load_data(train_path, X_train, y_train)

# Load testing data
print("Loading testing data...")
load_data(test_path, X_test, y_test)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Print shapes
print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTraining Labels Shape:")
print(y_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTesting Labels Shape:")
print(y_test.shape)