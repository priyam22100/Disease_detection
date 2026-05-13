import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Chest CT-Scan Classifier", page_icon="🫁", layout="wide")

st.title("🫁 Chest CT-Scan Classification with Grad-CAM")
st.write("Upload a Chest CT-Scan image to detect the type of disease and visualize the areas the model focused on.")

@st.cache_resource
def load_trained_model():
    return load_model("best_model.h5", compile=False)

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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Retrieve the inner model if wrapped
    target_model = model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                _ = layer.get_layer(last_conv_layer_name)
                target_model = layer
                break
            except:
                pass

    grad_model = tf.keras.models.Model(
        target_model.inputs,
        [target_model.get_layer(last_conv_layer_name).output, target_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    jet = cv2.resize(jet, (img_cv.shape[1], img_cv.shape[0]))
    superimposed_img = cv2.addWeighted(jet, alpha, img_cv, 1 - alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert('RGB')

    with col1:
        st.image(image, caption='Original Image', use_container_width=True)

    st.write("Analyzing the image...")

    # Needs to match the input shape of the loaded model dynamically
    model_input_shape = model.layers[0].input_shape[0][1:3] if isinstance(model.layers[0].input_shape, list) else model.layers[0].input_shape[1:3]
    if model_input_shape[0] is None:
        model_input_shape = (300, 300) # fallback

    img = image.resize(model_input_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_label = class_names[predicted_class_index]

    # More robust logic to find the last Conv2D layer specifically:
    last_conv_layer_name = None
    target_model = model

    # If the model is a Sequential or Functional wrapped around a base model (like EfficientNet)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            target_model = layer
            break

    # Now find the last convolutional layer in the target model
    for layer in reversed(target_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    # Specific fallback for EfficientNet models which use 'top_conv'
    if not last_conv_layer_name:
        try:
            target_model.get_layer('top_conv')
            last_conv_layer_name = 'top_conv'
        except:
            pass

    if last_conv_layer_name:
        try:
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_class_index)
            gradcam_img = display_gradcam(image.resize(model_input_shape), heatmap)
            with col2:
                st.image(gradcam_img, caption=f'Grad-CAM (Focus Area)', use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate Grad-CAM heatmap: {e}")
    else:
        st.warning("Could not find a convolutional layer for Grad-CAM.")

    st.markdown("### Prediction Result")

    if predicted_label == 'Normal':
        st.success(f"**Diagnosis:** {predicted_label}")
    else:
        st.error(f"**Diagnosis:** {predicted_label}")

    st.info(f"**Confidence:** {confidence:.2f}%")

    st.markdown("#### Detailed Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"- **{class_name}:** {predictions[0][i]*100:.2f}%")
