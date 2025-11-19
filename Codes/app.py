from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and preprocessing objects
try:
    model = joblib.load('svm_child_asd_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    print("Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    label_encoders = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        # Get form data
        data = request.json
        
        # Extract features in the correct order
        features = [
            float(data.get('A1_Score', 0)),
            float(data.get('A2_Score', 0)),
            float(data.get('A3_Score', 0)),
            float(data.get('A4_Score', 0)),
            float(data.get('A5_Score', 0)),
            float(data.get('A6_Score', 0)),
            float(data.get('A7_Score', 0)),
            float(data.get('A8_Score', 0)),
            float(data.get('A9_Score', 0)),
            float(data.get('A10_Score', 0)),
            float(data.get('age', 0)),
            float(data.get('gender', 0)),  # Assuming encoded: 0=Female, 1=Male
            float(data.get('jundice', 0)),  # 0=No, 1=Yes
            float(data.get('austim', 0)),   # 0=No, 1=Yes
            float(data.get('used_app_before', 0)),  # 0=No, 1=Yes
            float(data.get('result', 0))    # Sum of A1-A10 scores
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Convert prediction to readable format
        result = "ASD Traits Detected" if prediction == 1 else "No ASD Traits Detected"
        confidence = max(prediction_proba) * 100
        
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2f}%",
            'probability_asd': f"{prediction_proba[1]*100:.2f}%",
            'probability_no_asd': f"{prediction_proba[0]*100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)