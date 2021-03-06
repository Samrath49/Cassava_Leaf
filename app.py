import numpy as np
import pickle
import pandas as pd
# from flasgger import Swagger
import streamlit as st

from PIL import Image

# app=Flask(__name__)
# Swagger(app)

# pickle_in = open("classifier.pkl", "rb")
# classifier = pickle.load(pickle_in)

# @app.route('/')


# @app.route('/predict',methods=["Get"])


def predict_note_authentication(variance, skewness, curtosis, entropy):
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return prediction


def main():
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

    file = st.file_uploader('Upload An Image')
    dtype_file_structure_mapping = {
        'All Images': 'consolidated', 'Images Used To Train The Model': 'train',
        'Images Used To Tune The Model': 'valid', 'Images The Model Has Never Seen': 'test'
    }
    
    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("CSM Value", "Type Here")
    skewness = st.text_input("CGMP Value", "Type Here")
    curtosis = st.text_input("KGP Value", "Type Here")
    entropy = st.text_input("CDP Value", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(
            variance, skewness, curtosis, entropy)
    st.success('The disease is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
