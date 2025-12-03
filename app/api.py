from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import FeatureEngineer

app = FastAPI(title="Churn Prediction API", version="1.0")

# 1. Load Model at Startup
try:
    model = joblib.load('models/model_xgb.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# 2. Define the Data Structure (Validation)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: str  # Yes/No
    Partner: str        # Yes/No
    Dependents: str     # Yes/No
    tenure: int
    PhoneService: str  
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 3. The Prediction Endpoint
@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert incoming JSON -> DataFrame
    input_dict = data.dict()
    
    # Pre-processing: Convert Yes/No strings to 1/0 for SeniorCitizen if needed
    if input_dict['SeniorCitizen'] == 'Yes':
        input_dict['SeniorCitizen'] = 1
    else:
        input_dict['SeniorCitizen'] = 0

    df = pd.DataFrame([input_dict])
    
    # Predict
    try:
        probability = model.predict_proba(df)[0][1]
        prediction = int(model.predict(df)[0])
        
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": float(probability),
            "risk_level": "Critical" if probability > 0.7 else "Safe"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 4. Health Check (For AWS)
@app.get("/")
def home():
    return {"message": "Churn API is Running"}