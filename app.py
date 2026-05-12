import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

st.set_page_config(page_title="Chest CT-Scan Classifier", page_icon="🫁", layout="centered")

st.title("🫁 Chest CT-Scan Classification")
st.write("Upload a Chest CT-Scan image to detect the type of disease.")

@st.cache_resource
def load_trained_model():
    return load_model("best_model.h5")

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'best_model.h5' exists in the directory.")
    st.stop()

class_names = [
    'Adenocarcinoma',
    'Large Cell Carcinoma',
    'Normal',
    'Squamous Cell Carcinoma'
]

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Analyzing the image...")

    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Note: no rescaling for EfficientNet

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_label = class_names[predicted_class_index]

    st.markdown("### Prediction Result")

    if predicted_label == 'Normal':
        st.success(f"**Diagnosis:** {predicted_label}")
    else:
        st.error(f"**Diagnosis:** {predicted_label}")

    st.info(f"**Confidence:** {confidence:.2f}%")

    st.markdown("#### Detailed Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"- **{class_name}:** {predictions[0][i]*100:.2f}%")
