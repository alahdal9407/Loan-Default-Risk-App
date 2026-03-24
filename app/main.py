from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Load model
model = joblib.load(Path(__file__).parent.parent / "loan_default_model.pkl")

# Initialize app
app = FastAPI(title="Loan Default Prediction API")

# Define input schema
class LoanApplication(BaseModel):
    loan_amnt: float = Field(gt=0)
    term: int = Field(gt=0)
    int_rate: float = Field(gt=0)
    installment: float = Field(gt=0)
    grade: int = Field(ge=0, le=6)
    emp_length: int = Field(ge=0)
    annual_inc: float = Field(ge=0)
    pymnt_plan: int = Field(ge=0, le=1)
    dti: float = Field(ge=0)
    delinq_2yrs: float = Field(ge=0)
    fico_range_low: float = Field(ge=300, le=850)
    fico_range_high: float = Field(ge=300, le=850)
    inq_last_6mths: float = Field(ge=0)
    open_acc: float = Field(ge=0)
    pub_rec: float = Field(ge=0)
    revol_bal: float = Field(ge=0)
    revol_util: float = Field(ge=0, le=100)
    total_acc: float = Field(ge=0)
    credit_history_years: int = Field(ge=0)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is running!"}

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "expected_features": model.n_features_in_
    }

# Prediction endpoint
@app.post("/predict")
def predict(loan: LoanApplication):
    try:
        input_data = pd.DataFrame([loan.dict()])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        return {
            "prediction": "Default" if prediction == 1 else "Fully Paid",
            "default_probability": round(float(probability), 4),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))