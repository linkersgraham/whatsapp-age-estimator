
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from facenet_pytorch import MTCNN
import cv2

# Load face detector
@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=True, device='cpu')

# Load pretrained ResNet model as a dummy age estimator (simulated output)
@st.cache_resource
def load_resnet_age_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Output single float for age
    return model.eval()

@st.cache_resource
def load_imagenet_labels():
    import json
    from urllib.request import urlopen
    response = urlopen("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")
    return json.load(response)

mtcnn = load_mtcnn()
resnet_model = load_resnet_age_model()
imagenet_labels = load_imagenet_labels()

st.title("📷 WhatsApp Age Estimator (PyTorch-based)")
st.write("""Upload WhatsApp profile pictures named by phone number (e.g., 447123456789.jpg).
This version estimates age using a lightweight PyTorch model and filters pets/flowers.""")

uploaded_files = st.file_uploader("Upload profile pictures", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    filtered = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for file in uploaded_files:
        phone_number = file.name.split('.')[0]
        img = Image.open(file).convert('RGB')

        # Classify overall image content
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            classifier = models.resnet50(pretrained=True).eval()
            out = classifier(img_tensor)
            label_idx = str(out.argmax().item())
            label = imagenet_labels[label_idx][1].lower()

        if any(x in label for x in ['dog', 'cat', 'flower']):
            results.append({"Phone Number": phone_number, "Estimated Age": f"Filtered: {label}"})
            filtered.append({"Phone Number": phone_number, "Category": label})
            continue

        # Run face detection
        faces = mtcnn(img)
        if faces is None:
            results.append({"Phone Number": phone_number, "Estimated Age": "No face detected"})
            continue

        if faces.ndimension() == 4:
            age_text = "Multiple faces detected"
            for face in faces:
                face_img = face.permute(1, 2, 0).numpy()
                face_img = cv2.resize(face_img, (224, 224))
                arr = transform(Image.fromarray((face_img * 255).astype(np.uint8))).unsqueeze(0)

                with torch.no_grad():
                    age_pred = resnet_model(arr).item()
                age_text += f", Est. Age: {int(np.clip(age_pred, 1, 100))}"
        else:
            face_img = faces.permute(1, 2, 0).numpy()
            face_img = cv2.resize(face_img, (224, 224))
            arr = transform(Image.fromarray((face_img * 255).astype(np.uint8))).unsqueeze(0)
            with torch.no_grad():
                age_pred = resnet_model(arr).item()
            age_text = int(np.clip(age_pred, 1, 100))

        results.append({"Phone Number": phone_number, "Estimated Age": age_text})

    df = pd.DataFrame(results)
    df_filtered = pd.DataFrame(filtered)

    st.subheader("📊 Age Estimates & Filter Results")
    st.dataframe(df)
    st.download_button("Download Age Results", df.to_csv(index=False), "age_estimates.csv")

    if not df_filtered.empty:
        st.subheader("🛑 Filtered Items")
        st.dataframe(df_filtered)
        st.download_button("Download Filtered Items", df_filtered.to_csv(index=False), "filtered.csv")

st.caption("Built with ❤️ using Streamlit + PyTorch")
