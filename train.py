import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import kagglehub

KAGGLE_DATASET = "rajumavinmar/finger-print-based-blood-group-dataset"
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
label_map = {group: idx for idx, group in enumerate(blood_groups)}
IMG_SIZE = (128, 128)

print(f"1. Downloading dataset: {KAGGLE_DATASET}...")
try:
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"   Download successful. Path: {download_path}")
    dataset_path = os.path.join(download_path, "dataset")
    if not os.path.exists(dataset_path):
        dataset_path = download_path
    print(f"   Using base path for images: {dataset_path}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please ensure you have configured Kaggle API credentials if running locally, or that the environment has access.")
    raise

images = []
labels = []
print("2. Loading and preprocessing images...")

for group in blood_groups:
    folder_path = os.path.join(dataset_path, group)
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist. Skipping.")
        continue
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label_map[group])
        else:
            print(f"Warning: Could not load image {img_path}")

if len(images) == 0:
    raise ValueError("No images were loaded. Check the dataset path and contents.")

images = np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0
labels = to_categorical(np.array(labels), num_classes=len(blood_groups))
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f"   Total images loaded: {len(images)}")
print(f"   Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

print("3. Building and compiling CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(blood_groups), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("4. Starting model training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

print("5. Training completed. Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

model.save("blood_group_model.h5")
print("\nModel saved successfully as 'blood_group_model.h5'")