import os
import cv2
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_path = "dataset/train"
test_path = "dataset/test"

IMG_SIZE = 224

X_train = []
y_train = []

X_test = []
y_test = []

# Load dataset
def load_data(folder_path, X, y):

    for label in os.listdir(folder_path):

        label_path = os.path.join(folder_path, label)

        if os.path.isdir(label_path):

            for img_name in os.listdir(label_path):

                img_path = os.path.join(label_path, img_name)

                try:
                    img = cv2.imread(img_path)

                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                    img = img / 255.0

                    X.append(img)
                    y.append(int(label))

                except:
                    print("Error loading image")

print("Loading training data...")
load_data(train_path, X_train, y_train)

print("Loading testing data...")
load_data(test_path, X_test, y_test)

# Convert to arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# One-hot encoding
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

print("Dataset Loaded")

# Load MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base layers
base_model.trainable = True
optimizer=Adam(learning_rate=0.00001)

# Build model
model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(4, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()
datagen = ImageDataGenerator(

    rotation_range=15,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True
)

datagen.fit(X_train)
# Train model
history = model.fit(

    datagen.flow(X_train, y_train, batch_size=32),

    validation_data=(X_test, y_test),

    epochs=1
)

# Save model
model.save("improved_model.keras", include_optimizer=False)

print("\nImproved Model Saved Successfully")