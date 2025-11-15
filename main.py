import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.applications import ResNet50V2, MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

# Define dataset path (Adjust according to your Kaggle dataset directory)
DATASET_PATH = 'dataset-resized'

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100  # Increase epochs for thorough training
INITIAL_LEARNING_RATE = 0.0001  # Slightly lower initial learning rate for fine-tuning

# Load all image file paths and labels
image_paths = []
labels = []

# Get all class folders
for class_name in sorted(os.listdir(DATASET_PATH)):
    class_folder = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_folder):
        for image_file in os.listdir(class_folder):
            image_paths.append(os.path.join(class_folder, image_file))
            labels.append(class_name)

# Create a DataFrame
df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})

# Split dataset into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['label'], random_state=42)

# Image data generators with aggressive augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='image_path', y_col='label', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='image_path', y_col='label', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(test_df, x_col='image_path', y_col='label', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Build the hybrid model using ResNet50V2 and MobileNetV2
input_layer = Input(shape=(224, 224, 3))

# ResNet50V2
resnet_base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_base.trainable = True  # Fine-tune ResNet
resnet_features = resnet_base(input_layer)
resnet_features = GlobalAveragePooling2D()(resnet_features)

# MobileNetV2
mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_base.trainable = True  # Fine-tune MobileNetV2
mobilenet_features = mobilenet_base(input_layer)
mobilenet_features = GlobalAveragePooling2D()(mobilenet_features)

# Combine the two models' outputs
combined_features = tf.keras.layers.concatenate([resnet_features, mobilenet_features])

x = Dense(2048, activation='relu', kernel_regularizer=l2(0.01))(combined_features)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# Learning rate schedule using exponential decay
lr_schedule = ExponentialDecay(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

# Compile the model with exponential decay learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model in Keras format
model.save('model_resnet_mobilenet_99_accuracy_optimized.keras')

# Plot loss and accuracy graphs
plt.figure(figsize=(12, 6))

# Loss graph
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy graph
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
