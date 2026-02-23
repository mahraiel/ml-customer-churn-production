from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Load trained model
model = joblib.load("churn_model.pkl")


@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    """
    Expects JSON input with customer features
    """
    
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }
