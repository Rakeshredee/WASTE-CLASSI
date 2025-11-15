import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Ensure evaluation folder exists
os.makedirs("evaluation", exist_ok=True)

# Load trained model
model_path = "model_resnet_mobilenet_99_accuracy_optimized_final_100_epochs.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠️ Model file '{model_path}' not found. Train the model first.")

model = tf.keras.models.load_model(model_path)

# Load validation data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, validation_split=0.3)
val_data = datagen.flow_from_directory("dataset-resized", target_size=(224, 224), batch_size=16, subset="validation", shuffle=False)

# Class indices mapping
class_labels = list(val_data.class_indices.keys())

# Predict on validation data
y_val_pred = model.predict(val_data)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true = val_data.classes

# Generate classification report
val_report = classification_report(y_val_true, y_val_pred_classes, target_names=class_labels)
print("Validation Classification Report:\n", val_report)

# Save classification report
with open("evaluation/classification_report_validation.txt", "w") as f:
    f.write(val_report)

# Compute confusion matrix
val_cm = confusion_matrix(y_val_true, y_val_pred_classes)

# Compute accuracy
val_accuracy = np.mean(y_val_pred_classes == y_val_true)
print(f"Validation Accuracy: {val_accuracy:.2%}")

# Save accuracy
with open("evaluation/accuracy.txt", "w") as f:
    f.write(f"Validation Accuracy: {val_accuracy:.2%}\n")
