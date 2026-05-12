import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import sys
from pathlib import Path

# Add the current directory to path to import train_deepfake_detector
sys.path.insert(0, str(Path(__file__).parent))
from train_deepfake_detector import build_model

# Load model
@st.cache_resource
def load_model():
    checkpoint = torch.load('models/deepfake_detector_best.pt', map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    model_name = checkpoint.get('model_name', 'mobilenet')
    
    model = build_model(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

model, config = load_model()
frames_per_video = config.get('frames_per_video', 8)
image_size = config.get('image_size', 128)

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    # Convert PIL to OpenCV
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        st.error("No face detected in the image.")
        return None
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_pil = Image.fromarray(face)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(face_pil)
    # Repeat to frames_per_video frames
    tensor = tensor.unsqueeze(0).repeat(frames_per_video, 1, 1, 1).unsqueeze(0)  # (1, frames_per_video, 3, image_size, image_size)
    return tensor

st.title("Deepfake Detector")
st.write("Upload an image to classify it as real or deepfake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Classify"):
        with st.spinner("Processing..."):
            input_tensor = preprocess_image(image)
            if input_tensor is not None:
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1).squeeze(0)
                    predicted_class = int(probabilities.argmax().item())
                    confidence = float(probabilities[predicted_class].item()) * 100
                    class_names = ['Real', 'Fake']
                    label = class_names[predicted_class]
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {confidence:.2f}%")