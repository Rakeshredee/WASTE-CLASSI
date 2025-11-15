import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pyttsx3  # Voice output
from ultralytics import YOLO

# Paths to models
CLASSIFICATION_MODEL_PATH = "model_resnet_mobilenet_99_accuracy_optimized_final_100_epochs.keras"
YOLO_MODEL_PATH = "best.pt"
IMAGE_SIZE = (224, 224)

# Load models
@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)

@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

classification_model = load_classification_model()
yolo_model = load_yolo_model()

# Class labels
class_labels = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Text-to-speech function
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Classification function
def classify_object(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, IMAGE_SIZE)
    img_resized = np.expand_dims(img_resized / 255.0, axis=0)
    predictions = classification_model.predict(img_resized)
    class_idx = np.argmax(predictions)
    return class_labels[class_idx]

# Object detection function (Only for webcam)
def detect_objects(img, confidence_threshold=0.5):
    results = yolo_model(img)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            if conf >= confidence_threshold:
                detections.append((x1, y1, x2, y2))
    return detections

# Streamlit UI
st.title("Waste Classification & Object Detection")

option = st.radio("Select Input Method:", ["Upload Image", "Use Webcam"])
detected_class = ""

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        detected_class = classify_object(img_np)  # Direct classification

        # Display image in a small box
        st.markdown("<div style='border: 2px solid black; padding: 10px; display: inline-block;'>", unsafe_allow_html=True)
        st.image(img_np, caption="Uploaded Image", use_column_width=False, width=250)
        st.markdown("</div>", unsafe_allow_html=True)

        # Display predicted text in BIG RED font
        st.markdown(f"<h3 style='color: red;'>Predicted Class = {detected_class}</h3>", unsafe_allow_html=True)

        # Speak detected class
        speak_text(f"Detected: {detected_class}")

        if st.button("Repeat Detected Object"):
            speak_text(f"Detected: {detected_class}")

elif option == "Use Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)

        for x1, y1, x2, y2 in detections:
            obj = frame[y1:y2, x1:x2]  # Extract object
            category = classify_object(obj)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
