
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from facenet_pytorch import MTCNN
import torchvision.models as models
import torch
import torchvision.transforms as transforms

# Load DEX model
@st.cache_resource
def load_model():
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(101, activation='softmax'))
    model.load_weights('dex_weights.h5')
    return model

# Load face detector
@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=True, device='cpu')

# Load image classifier for content filtering
@st.cache_resource
def load_resnet():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# Load labels for ImageNet
@st.cache_resource
def load_imagenet_labels():
    import json
    from urllib.request import urlopen
    response = urlopen("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")
    return json.load(response)

model = load_model()
mtcnn = load_mtcnn()
resnet = load_resnet()
imagenet_labels = load_imagenet_labels()

st.title("📷 WhatsApp Profile Age Estimator + Smart Filter")
st.write("Upload WhatsApp profile pictures named by phone number (e.g., 447123456789.jpg). This app will crop the face and estimate age using a deep learning model. It will filter out pets, flowers, and non-human images.")

uploaded_files = st.file_uploader("Upload profile pictures", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    ages = np.arange(0, 101).reshape(101, 1)
    results = []
    filtered = []

    for uploaded_file in uploaded_files:
        phone_number = uploaded_file.name.split('.')[0]
        image = Image.open(uploaded_file).convert('RGB')

        # Filter image categories (dog, cat, flower detection)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = resnet(img_tensor)
            _, pred = outputs.max(1)
            class_id = str(pred.item())
            class_label = imagenet_labels[class_id][1].lower()

        if any(animal in class_label for animal in ["dog", "cat", "flower"]):
            results.append({"Phone Number": phone_number, "Estimated Age": "Filtered: " + class_label})
            filtered.append({"Phone Number": phone_number, "Category": class_label})
            continue

        # Face detection and age estimation
        try:
            faces = mtcnn(image)
            if faces is not None and faces.ndimension() == 4 and len(faces) > 1:
                age_text = "Multiple faces detected"
                for face in faces:
                    face = face.permute(1, 2, 0).numpy()
                    face = cv2.resize(face, (224, 224))
                    arr = img_to_array(face) / 255.0
                    arr = np.expand_dims(arr, axis=0)

                    prediction = model.predict(arr)[0]
                    predicted_age = int(np.round(np.sum(prediction * ages)))
                    age_text += f", Est. Age: {predicted_age}"
            elif faces is not None:
                face = faces[0].permute(1, 2, 0).numpy()
                face = cv2.resize(face, (224, 224))
                arr = img_to_array(face) / 255.0
                arr = np.expand_dims(arr, axis=0)

                prediction = model.predict(arr)[0]
                predicted_age = int(np.round(np.sum(prediction * ages)))
                age_text = predicted_age
            else:
                age_text = "No face detected"

        except Exception as e:
            age_text = f"Error: {e}"

        results.append({"Phone Number": phone_number, "Estimated Age": age_text})

    df = pd.DataFrame(results)
    df_filtered = pd.DataFrame(filtered)

    st.subheader("📊 Age Estimates & Filter Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Age Results as CSV", csv, "age_estimates.csv", "text/csv")

    if not df_filtered.empty:
        st.subheader("🛑 Filtered Images (Pets & Flowers)")
        st.dataframe(df_filtered)
        csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Items as CSV", csv_filtered, "filtered_images.csv", "text/csv")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, DEX, ResNet, and MTCNN")
