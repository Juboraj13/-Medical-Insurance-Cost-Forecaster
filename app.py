from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Medical Insurance Cost Forecaster API",
    description="Production-ready model endpoint generating custom annual health premium pricing models."
)

class CustomerProfile(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

model_pipeline = joblib.load('best_insurance_model.pkl')

@app.get("/")
def health_check():
    return {"status": "Operational", "engine": "RandomForestRegressor"}

@app.post("/predict")
def generate_forecast(profile: CustomerProfile):
    input_data = pd.DataFrame([{
        'age': profile.age,
        'sex': profile.sex,
        'bmi': profile.bmi,
        'children': profile.children,
        'smoker': profile.smoker,
        'region': profile.region
    }])
    
    raw_prediction = model_pipeline.predict(input_data)[0]
    
    return {
        "status": "Success",
        "predicted_annual_charges_usd": round(float(raw_prediction), 2)
    }
