from typing import Union
import numpy as np
from fastapi import FastAPI

from model import PredictionRequest
import pickle
app = FastAPI()


@app.on_event("startup")
def load_model():
    global model, le_home, le_status, scaler
    with open('Gradient Boosting_model.pkl', 'rb') as file:
        model = pickle.load(file)  # Load your ML model

    with open('label_encoder_home.pkl', 'rb') as file:
        le_home = pickle.load(file)  # Load home label encoder

    
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file) 


        
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Convert request data to a list
    data = [[
        request.home,
        request.status,
        request.amount,
        request.emp_length,
        request.rate,
        request.percent_income
    ]]
    
    # Encode categorical variables
    data[0][0] = le_home.transform([request.home])[0]  # Transform 'home'

    # Convert to numpy array for scaling
    data = np.array(data, dtype=float)
    # Scale the numerical features
    data_scaled = scaler.transform(data)

    # Make the prediction
    prediction = model.predict(data_scaled)

    # Convert prediction result to a human-readable response
    result = "Default" if prediction[0] == 1 else "No Default"

    return {"prediction": result}