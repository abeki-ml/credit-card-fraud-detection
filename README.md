Credit Card Fraud Detection

A machine learning project that detects potentially fraudulent credit card transactions using Logistic Regression.
Technologies

    Python
    Pandas and NumPy
    Scikit-learn
    Joblib
    FastAPI

Model Performance

    Fraud Recall: 86.5%
    Fraud Precision: 13.6%
    F1-Score: 23.5%
    Decision Threshold: 0.7

The model prioritizes detecting fraudulent transactions while balancing false alarms.
Features

    Handles imbalanced data using class weighting.
    Uses a machine learning pipeline for preprocessing and prediction.
    Provides a REST API for fraud predictions.
    Supports probability-based classification with a configurable threshold.

Run the API

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn src.api:app --reload

Open http://127.0.0.1:8000/docs to test the API.
Dataset

The dataset is not included in this repository because of its size. Obtain the Credit Card Fraud Detection dataset on Kaggle and place creditcard.csv in the data/ directory.

Note: The model's precision means some legitimate transactions may be flagged as potentially fraudulent.