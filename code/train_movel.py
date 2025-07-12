import os
import random
from shutil import copy2

import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Define directories
base_dir = 'path_to_dataset'           # Replace with your base dataset path
train_dir = 'path_to_train_dataset'    # Replace with your train dir path
test_dir = 'path_to_test_dataset'      # Replace with your test dir path

categories = ['Healthy', 'Unhealthy']

# Create category subdirectories for train and test
for cat in categories:
    os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cat), exist_ok=True)

def process_images(category):
    """Split images from base_dir into train and test folders."""
    source_dir = os.path.join(base_dir, category)
    if not os.path.isdir(source_dir):
        print(f"Directory not found: {source_dir}")
        return

    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    if not images:
        print(f"No images found in {source_dir}")
        return

    train_imgs, test_imgs = train_test_split(images, test_size=0.20, random_state=42)

    for img in train_imgs:
        copy2(os.path.join(source_dir, img), os.path.join(train_dir, category, img))
    for img in test_imgs:
        copy2(os.path.join(source_dir, img), os.path.join(test_dir, category, img))

    print(f"Processed {len(train_imgs)} training and {len(test_imgs)} testing images in category '{category}'.")

# Process both categories
for category in categories:
    process_images(category)

# Data augmentation for training; rescaling for testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.6),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=153,  # Adjust depending on dataset size
    epochs=8,
    validation_data=validation_generator,
    validation_steps=38,  # Adjust depending on dataset size
    callbacks=[early_stopping, reduce_lr]
)
