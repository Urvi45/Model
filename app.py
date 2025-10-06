from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Decision Tree Flight Price Predictor is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
