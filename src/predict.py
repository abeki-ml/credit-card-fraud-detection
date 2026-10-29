import pandas as pd

import joblib

data = joblib.load("model/fraud_pipeline.pkl")

model = data["pipeline"]
threshold = data["threshold"]
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns="Class")
transaction = X.iloc[[0]]
probability = model.predict_proba(transaction)[0,1]

prediction =(probability >=threshold).astype(int)
if prediction == 1:
    print("Prediction: Potential Fraud")
else:
    print("Prediction: Not Flagged as Fraud")