from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("../models/eta_model.pkl")

@app.post("/predict_eta")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    prediction = model.predict(df)[0]

    return {"eta": float(prediction)}