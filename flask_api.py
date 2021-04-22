from flask import Flask, request
import numpy as np
from keras.models import load_model
import h5py
import pandas as pd
import flasgger
from flasgger import Swagger
from PIL import Image, ImageOps

app = Flask(__name__)
Swagger(PathoAI)

model_in = open("cassava.hdf5", "rb")
model = load_model(model_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict', methods=["Get"])
def predict_note_authentication():

    file = request.files['image']
    size = (512, 512)
    img = np.asarray(file)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    print(prediction)
    return "Hello The answer is"+str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = model.predict(df_test)

    return str(list(prediction))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
