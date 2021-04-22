from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import pickle
import pandas as pd
import streamlit as st

st.set_options('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.model.load_model('./cassava.hdf5')
    return model


model = load_model()


def import_and_predict(image_data, model):
    size = (512, 512)
    image = ImageOps.fir(image_data, size, Image_ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


st.title("PathoAI")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Cassava Leaf Disease Detection ML App</h2>
</div>
"""

instructions = """
Either upload your own image or select from the sidebar to get a preconfigured image. 
The image you select or upload will be fed through the Deep Neural Network in real-time 
and the output will be displayed to the screen.
    """
st.write(instructions)


file = st.file_uploader('Upload An Image', type=['jpg', 'png'])
st.title("Here is the image you've selected")

st.markdown(html_temp, unsafe_allow_html=True)

result = ""
if file is None:
    st.text("Please upload an image")

elif st.button("Predict"):
    image = Image.open(file)
    st.image(image, use_column_width=True)
    result = import_and_predict(image, model)
    st.success('The output is {}'.format(result))


if st.button("About"):
    st.text("Cassava Lead Disease Detection Application")
    st.text("Developed by AiBoost")
