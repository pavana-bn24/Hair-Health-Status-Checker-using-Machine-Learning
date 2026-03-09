import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# --- compatibility fix for old sklearn model ---
import sklearn.linear_model._base
import sys
sys.modules['sklearn.linear_model.base'] = sklearn.linear_model._base
# -----------------------------------------------

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # get input values
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    # convert prediction to percentage-like value
    output = abs(int(prediction[0]) % 100)

    # classify risk level
    if output < 30:
        result = "Hair Health Status: Strong Hair (Low Risk) 🟢"
    elif output < 60:
        result = "Hair Health Status: Moderate Hair Fall Risk 🟡"
    else:
        result = "Hair Health Status: High Hair Fall Risk 🔴"

    return render_template('index.html', prediction_text=result)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = abs(int(prediction[0]) % 100)

    if output < 30:
        result = "Low Hair Fall Risk"
    elif output < 60:
        result = "Moderate Hair Fall Risk"
    else:
        result = "High Hair Fall Risk"

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)