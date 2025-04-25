
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN
import torch.nn as nn
import os

# Define the UTKFace Age Estimator model (ResNet-18 for regression)
class AgeEstimator(nn.Module):
    def __init__(self):
        super(AgeEstimator, self).__init__()
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

# Load face detector and pretrained model
@st.cache_resource
def load_model_and_detector():
    mtcnn = MTCNN(keep_all=True, device='cpu')
    model = AgeEstimator()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/age_model_resnet18_utkface.pth',
        map_location='cpu'
    ))
    model.eval()
    return mtcnn, model

mtcnn, model = load_model_and_detector()

st.title("🎯 WhatsApp Age Estimator (UTKFace Powered)")
st.write("Upload WhatsApp profile pictures named by phone number (e.g., 447123456789.jpg). Now using a pretrained UTKFace model for real age prediction!")

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

        # Classify general content to filter cats, dogs, flowers
        with torch.no_grad():
            classifier = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True).eval()
            labels = torch.hub.load('pytorch/vision', 'imagenet_classes')
            out = classifier(transform(img).unsqueeze(0))
            label = labels[out.argmax().item()].lower()

        if any(x in label for x in ['dog', 'cat', 'flower']):
            results.append({"Phone Number": phone_number, "Estimated Age": f"Filtered: {label}"})
            filtered.append({"Phone Number": phone_number, "Category": label})
            continue

        # Detect faces
        faces = mtcnn(img)
        if faces is None:
            results.append({"Phone Number": phone_number, "Estimated Age": "No face detected"})
            continue

        ages = []
        if faces.ndim == 4:
            for face in faces:
                pil_face = Image.fromarray((face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((224, 224))
                input_tensor = transform(pil_face).unsqueeze(0)
                with torch.no_grad():
                    pred = model(input_tensor)
                ages.append(str(int(np.clip(pred.item(), 1, 100))))
            age_text = "Multiple faces detected, Est. Age(s): " + ", ".join(ages)
        else:
            pil_face = Image.fromarray((faces.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((224, 224))
            input_tensor = transform(pil_face).unsqueeze(0)
            with torch.no_grad():
                pred = model(input_tensor)
            age_text = int(np.clip(pred.item(), 1, 100))

        results.append({"Phone Number": phone_number, "Estimated Age": age_text})

    df = pd.DataFrame(results)
    df_filtered = pd.DataFrame(filtered)

    st.subheader("📊 Age Estimates")
    st.dataframe(df)
    st.download_button("Download Age Results", df.to_csv(index=False), "age_estimates.csv")

    if not df_filtered.empty:
        st.subheader("🛑 Filtered Images")
        st.dataframe(df_filtered)
        st.download_button("Download Filtered Items", df_filtered.to_csv(index=False), "filtered.csv")

st.caption("Built with ❤️ using Streamlit + PyTorch + UTKFace")
