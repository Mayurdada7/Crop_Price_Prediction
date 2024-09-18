from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and encoders
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract data
    crop_type = data['crop_type']
    city = data['city']
    season = data['season']
    temperature = data['temperature']

    # Process input
    encoded_crop = label_encoders['Crop_Type'].transform([crop_type])[0]
    encoded_city = label_encoders['City'].transform([city])[0]
    encoded_season = label_encoders['Season'].transform([season])[0]
    temperature = scaler.transform([[temperature]])[0][0]

    # Prepare input for prediction
    input_data = pd.DataFrame([[encoded_crop, encoded_city, encoded_season, temperature]], 
                              columns=['Crop_Type', 'City', 'Season', 'Temperature'])

    # Make prediction
    prediction = xgb_model.predict(input_data)[0]

    # Convert prediction to standard Python float and round to 2 decimal places
    prediction = round(float(prediction), 2)

    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)
