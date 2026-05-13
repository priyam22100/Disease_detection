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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert RGB image to BGR for cv2
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Resize heatmap to match image size
    jet = cv2.resize(jet, (img_cv.shape[1], img_cv.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(jet, alpha, img_cv, 1 - alpha, 0)

    # Convert BGR back to RGB for display in Streamlit
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 2 columns layout
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert('RGB')

    with col1:
        st.image(image, caption='Original Image', use_column_width=True)

    st.write("Analyzing the image...")

    # Needs to match train.py IMG_SIZE (300, 300)
    img = image.resize((300, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_label = class_names[predicted_class_index]

    # Find last conv layer for Grad-CAM
    # EfficientNetB3 structure has a 'top_conv' layer, let's find the last Conv2D dynamically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        # We look into the base_model if our model is wrapped
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if len(inner_layer.output_shape) == 4:
                    last_conv_layer_name = inner_layer.name
                    model = layer # Re-assign model to base_model for gradcam to work properly
                    break
            break
        elif len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name:
        try:
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_class_index)
            gradcam_img = display_gradcam(image.resize((300, 300)), heatmap)
            with col2:
                st.image(gradcam_img, caption=f'Grad-CAM (Focus Area)', use_column_width=True)
        except Exception as e:
            st.warning("Could not generate Grad-CAM heatmap for this model architecture.")
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
