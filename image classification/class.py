# Import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

print('All imports were successful!')

# Load the butterfly dataset
df_training = pd.read_csv("ML\image classification\Training_set.csv")
df_testing = pd.read_csv("ML\image classification\Testing_set.csv")

# Split the training dataset into training and validation sets
train_data, val_data = train_test_split(df_training, test_size=0.2, random_state=42)

# Define image directory path
image_dir = r'ML\image classification\train'

# Data augmentation for training and rescaling for validation
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

val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data with ImageDataGenerator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
# Create a grid for displaying butterflies
fig4, axes = plt.subplots(2, 4, figsize=(15, 8))

# Randomly sample
sample_images = df_training.sample(8, random_state=42)

# Display images
for i, (index, row) in enumerate(sample_images.iterrows()):
    image_path = os.path.join(image_dir, row['filename'])
    image = load_img(image_path, target_size=(150, 150))
    image_array = img_to_array(image) / 255.0 # For normalization
    ax = axes[i // 4, i % 4]
    
    ax.imshow(image_array)
    ax.set_title(f"Class: {row['label']}")
    ax.axis('off')  # Hide axes

# Show images
plt.tight_layout()
plt.show()

# Create another table and visualize number of butterflies by their number
class_counts = df_training['label'].value_counts().sort_index()
fig1 = plt.figure(figsize=(20, 10))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='plasma')
plt.title('Number of Butterflies by Their Number')
plt.xlabel('Butterfly Classes')
plt.ylabel('Number of Butterflies')
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.show()

# Build a neural network model
model_NN = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(75, activation='softmax')
])

# Compile the model
model_NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary the model
model_NN.summary()

# Train the NN model with 40 epochs and get the highest accuracy value
history = model_NN.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=40,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Plot the training and validation accuracy and loss
plt.figure(figsize=(20, 10))

# First plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Second plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Show plots
plt.tight_layout()
plt.show()

# Evaluate the model performance on validation data
val_loss, val_acc = model_NN.evaluate(val_generator)
print(f'Validation Accuracy: {val_acc}')
print(f'Validation Loss: {val_loss}')