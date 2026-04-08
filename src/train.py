import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw", "deliveries.csv")

df = pd.read_csv(data_path)
df = df.dropna()

# Encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("delivery_time", axis=1)
y = df["delivery_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"MAE: {mae:.2f} minutos")

# 🔥 Guardar columnas (CLAVE)
joblib.dump(X.columns.tolist(), os.path.join(BASE_DIR, "models", "features.pkl"))

# Guardar modelo
joblib.dump(model, os.path.join(BASE_DIR, "models", "eta_model.pkl"))