import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../models/eta_model.pkl")

st.title("OpsVision AI")
st.subheader("Predicción de tiempo de entrega")

distance = st.slider("Distancia (km)", 1.0, 20.0)
traffic = st.selectbox("Tráfico", ["low", "medium", "high"])
weather = st.selectbox("Clima", ["clear", "rain", "storm"])

input_data = pd.DataFrame([{
    "distance": distance,
    "traffic": traffic,
    "weather": weather
}])

input_data = pd.get_dummies(input_data)

if st.button("Predecir ETA"):
    prediction = model.predict(input_data)[0]

    st.write(f"⏱️ Tiempo estimado: {prediction:.2f} minutos")

    if prediction > 40:
        st.error("⚠️ Alto riesgo de retraso")
    else:
        st.success("✅ Entrega en tiempo esperado")

st.markdown("### 📊 Factores clave")
st.write("- Tráfico alto impacta fuertemente el tiempo")
st.write("- Clima adverso aumenta retrasos")
st.write("- Distancia no es el factor principal")