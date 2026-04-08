import joblib
import pandas as pd

model = joblib.load("../models/eta_model.pkl")

def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {
        "estimated_delivery_time": float(prediction)
    }