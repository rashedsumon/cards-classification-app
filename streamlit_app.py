import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from model import load_model
from data_loader import download_dataset

# -------------------
# Load dataset & model
# -------------------
data_path = download_dataset()

# assuming train folder exists
class_names = sorted(os.listdir(data_path + "/train"))

device = "cpu"
model = load_model(len(class_names), device=device)

# image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🃏 Playing Cards Classifier")

uploaded_file = st.file_uploader("Upload a card image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: {class_names[predicted.item()]}")