# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('gbc_scale_flask.pkl', 'rb') as f:
        model = pickle.load(f)

def load_scaler():
    global scaler
    with open('scale_flask.pkl', 'rb') as g:
        scaler = pickle.load(g)


@app.route('/')
def home_endpoint():
    return 'Hello World! This is a production server for Credit Risk Model.'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (8,) to (1, 8)
        data = scaler.transform(data) #scale new data

        np.set_printoptions(formatter={'float_kind':'{:f}'.format})

        prediction = model.predict(data)  # runs globally loaded model on the data
        probability= model.predict_proba(data)
    return str(prediction[0])

    return str('{:.2%}'.format(probability[0,0]))

    return str('{:.2%}'.format(probability[0,1]))



if __name__ == '__main__':
    load_model() # load model at the beginning once only
    load_scaler()#load scaler at the beginning once only
    app.run(host='0.0.0.0')
