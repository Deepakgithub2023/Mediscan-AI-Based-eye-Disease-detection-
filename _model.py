import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# Check GPU Availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Define Dataset Paths (Update paths as needed)
train_dir = r"unzip_data set 2\DATASET_101\Train"
test_dir = r"unzip_data set 2\DATASET_101\Test"

# Function to create DataFrame from image directory
def create_dataframe(image_dir):
    class_labels = []
    filenames = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                filenames.append(os.path.relpath(os.path.join(root, file), image_dir))
                class_labels.append(os.path.basename(os.path.dirname(os.path.join(root, file))))

    print(f"Found {len(filenames)} images in {image_dir}")
    print(f"Classes: {set(class_labels)}")
    return pd.DataFrame({'filename': filenames, 'class': class_labels})

# Print directory structure for debugging
def print_directory_contents(directory):
    for root, dirs, files in os.walk(directory):
        print(f"Root: {root}")
        print(f"Directories: {dirs}")
        print(f"Files: {files}")
        print("-" * 20)

print("Training Directory Contents:")
print_directory_contents(train_dir)

# Create DataFrame for training data
train_df = create_dataframe(train_dir)
train_df.columns = train_df.columns.str.strip()
print(train_df.head())

# Create ImageDataGenerator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of training data used for validation
)

# Train data loader
train_data = datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col="filename",
    y_col="class",
    subset="training",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Validation data loader
val_data = datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col="filename",
    y_col="class",
    subset="validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Get number of classes
num_classes = len(train_data.class_indices)

# Load VGG19 pretrained model
base_model = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze some layers
for layer in base_model.layers[:15]:
    layer.trainable = False

# Build the full model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=callbacks
)

# Save trained model
model.save("vgg19_model_improved.h5")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Load test data
test_df = create_dataframe(test_dir)

test_data = datagen.flow_from_dataframe(
    test_df,
    directory=test_dir,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False  # Important if you want ordered predictions later
)

# Evaluate on test data
final_loss, final_accuracy = model.evaluate(test_data)
print(f"Final Test Loss: {final_loss:.4f}")
print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")

# Final training stats
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")
