import os
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

# Image size
IMG_SIZE = 224

# Data containers
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

                    # Store
                    X.append(img)
                    y.append(int(label))

                except:
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

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

print("Data Loaded Successfully")

# Build CNN model
model = Sequential()

# First convolution layer
model.add(Conv2D(32, (3,3), activation='relu',
                 input_shape=(224,224,3)))

model.add(MaxPooling2D(pool_size=(2,2)))

# Second convolution layer
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# Third convolution layer
model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten layer
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu'))

# Dropout to reduce overfitting
model.add(Dropout(0.5))

# Output layer
model.add(Dense(4, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model.h5")

print("\nModel Saved Successfully")