# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the pre-trained Pix2Pix generator model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model/pix2pix_generator.h5', compile=False)
    return model

# app.py (continued)

# Function to preprocess the input image
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)
    return image

# Function to postprocess the output image
def postprocess_image(image):
    image = (image + 1) * 127.5  # Denormalize to [0, 255]
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    image = Image.fromarray(image[0])
    return image

def main():
    st.title("🎨 Sketch to Color with Pix2Pix")
    st.write("Upload a sketch, and the model will generate a colorized version.")

    # Load the model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose a sketch image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.subheader("Original Sketch")
        st.image(image, use_column_width=True)

        # Preprocess the image
        input_image = preprocess_image(image)

        # Generate the colorized image
        with st.spinner('Generating colorized image...'):
            output_image = model.predict(input_image)
            output_image = postprocess_image(output_image)

        st.subheader("Colorized Image")
        st.image(output_image, use_column_width=True)

if __name__ == "__main__":
    main()
