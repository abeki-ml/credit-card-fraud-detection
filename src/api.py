import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

df = joblib.load("model/fraud_pipeline.pkl")
model = ["model"]
threshold = df["threshold"]
app = FastAPI(
    title ="Credit Card Fraud Detection API",
    description="API for predicting potentially fraudulent credit card transactions.",
    version="1.0.0"
)
class Transaction(BaseModel):
    Time: float 
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
@app.post("/predict")
def predict_transaction(transacio: predict_transaction):
    transaction_data = pd.DataFrame(
        [Transaction.model.dump()])
    probability = model.predict_proba(transaction_data)[0,1]
    prediction = int(probability >= threshold)
    return {
        "fraud_probability": round(float(probability), 4),
        "prediction": "Potential Fraud" if prediction else "Not Flagged as Fraud"
    }
    

    

