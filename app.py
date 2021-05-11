import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st

from PIL import Image

# app=Flask(__name__)
# Swagger(app)

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# @app.route('/')


def welcome():
    return "Welcome All"

# @app.route('/predict',methods=["Get"])


def predict_note_authentication(variance, skewness, curtosis, entropy):
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return prediction


def main():
    st.title("PathoAI")
    st.text("  ")

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
    st.text("  ")
    st.write(instructions)
    st.text("  ")

    file = st.file_uploader('Upload An Image')
    dtype_file_structure_mapping = {
        'All Images': 'consolidated', 'Images Used To Train The Model': 'train',
        'Images Used To Tune The Model': 'valid', 'Images The Model Has Never Seen': 'test'
    }

    st.image(file, caption="Uploaded Image")
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("Cassava Bacterial Blight(CBB)", "Type Here")
    skewness = st.text_input(
        "Cassava Brown Streak Disease (CBSD)", "Type Here")
    curtosis = st.text_input("Cassava Green Mottle (CGM)", "Type Here")
    entropy = st.text_input("Cassava Mosaic Disease (CMD)", "Type Here")
    result = ""

    st.text(" ")
    if st.button("Predict"):
        result = predict_note_authentication(
            variance, skewness, curtosis, entropy)
    st.success('Prediction is {}'.format(result))
    st.text(" ")
    if st.button("About"):
        st.text("PathoAI")
        st.text("Prediction 0 means it's a healthy leaf else it is unhealthy")


if __name__ == '__main__':
    main()
