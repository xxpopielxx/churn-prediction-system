import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

#SETUP PATHS & LOAD MODEL
current_dir = os.path.dirname(__file__)

model_path = os.path.join(current_dir, '..', 'models', 'churn_model.pkl')

print(f"Attempting to load model from: {model_path}...")

try:
    #Load the dictionary containing both the model and the feature list
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']
    print("Model loaded successfully. Ready to predict.")
except Exception as e:
    print(f"ERROR: Failed to load model. Check the path {e}")
    model = None
    features = []

#DEFINE INPUT DATA STRUCTURE 
class CustomerData(BaseModel):
    class Config:
        extra = "allow" 

#INITIALIZE API
app = FastAPI(title="Churn Prediction API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running"}

@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__ if model else "No model loaded",
        "features_count": len(features),
        "expected_features": features[:10]  # Show first 10 features as a sample
    }

@app.post("/predict")
def predict_churn(data: CustomerData):
    #Check if model is loaded
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        #1. Convert input JSON (Pydantic model) to a Python dictionary
        input_data = data.dict()

        #2. Convert dictionary to a Pandas DataFrame
        input_df = pd.DataFrame([input_data])

        #3. Align columns with the training data
        # The model expects specific columns in a specific order.
        # We use .reindex() to ensure the input matches the training features.
        # Missing columns (e.g., if user didn't send 'Gender_Male') are filled with 0.
        input_df = input_df.reindex(columns=features, fill_value=0)

        #4. Make Prediction
        # result will be [0] (No Churn) or [1] (Churn)
        prediction = model.predict(input_df)[0]
        
        # Get probability
        # result will be something like [0.25, 0.75]
        probability = model.predict_proba(input_df)[0][1]

        #5. Return the result as JSON
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "message": "Customer will leave" if prediction == 1 else "Customer will stay"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")