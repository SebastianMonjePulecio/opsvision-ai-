import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==============================
# 📂 CARGA DE DATOS
# ==============================

df = pd.read_csv("data/raw/deliveries.csv")

model = joblib.load("models/eta_model.pkl")
features = joblib.load("models/features.pkl")

st.set_page_config(page_title="OpsVision AI Dashboard", layout="wide")

# ==============================
# 🧠 HEADER
# ==============================

st.title("🚀 OpsVision AI - Dashboard Operativo")

st.markdown("""
## 📦 Delivery Operations Intelligence

Este dashboard analiza entregas para identificar retrasos, entender sus causas y predecir tiempos de entrega (ETA).
""")

st.markdown("---")

# ==============================
# 🎛️ FILTROS (NIVEL EMPRESA)
# ==============================

st.sidebar.header("Filtros")

traffic_filter = st.sidebar.multiselect(
    "Tráfico", df["traffic"].unique(), default=df["traffic"].unique()
)

weather_filter = st.sidebar.multiselect(
    "Clima", df["weather"].unique(), default=df["weather"].unique()
)

filtered_df = df[
    (df["traffic"].isin(traffic_filter)) &
    (df["weather"].isin(weather_filter))
]

# ==============================
# 📊 KPIs
# ==============================

avg_time = filtered_df["delivery_time"].mean()
max_time = filtered_df["delivery_time"].max()
min_time = filtered_df["delivery_time"].min()

col1, col2, col3 = st.columns(3)

col1.metric("⏱️ Tiempo Promedio", f"{avg_time:.2f} min")
col2.metric("📈 Tiempo Máximo", f"{max_time} min")
col3.metric("📉 Tiempo Mínimo", f"{min_time} min")

st.caption("El tiempo promedio refleja la eficiencia general de las entregas")

st.markdown("---")

# ==============================
# 🚗 TRÁFICO
# ==============================

st.subheader("🚗 Impacto del tráfico")

traffic_analysis = filtered_df.groupby("traffic")["delivery_time"].mean()

fig, ax = plt.subplots()
traffic_analysis.plot(kind="bar", ax=ax)
ax.set_title("Tiempo promedio por tráfico")
st.pyplot(fig)

st.markdown("---")

# ==============================
# 🌧️ CLIMA
# ==============================

st.subheader("🌧️ Impacto del clima")

weather_analysis = filtered_df.groupby("weather")["delivery_time"].mean()

fig, ax = plt.subplots()
weather_analysis.plot(kind="bar", ax=ax)
ax.set_title("Tiempo promedio por clima")
st.pyplot(fig)

st.markdown("---")

# ==============================
# ⚠️ ALERTAS
# ==============================

st.subheader("⚠️ Entregas con riesgo")

st.warning("Pedidos con alto riesgo pueden afectar la satisfacción del cliente y generar cancelaciones")

high_risk = filtered_df[filtered_df["delivery_time"] > 40]

risk_percentage = (len(high_risk) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0

col_risk1, col_risk2 = st.columns(2)

col_risk1.metric("Pedidos en riesgo", len(high_risk))
col_risk2.metric("⚠️ % en riesgo", f"{risk_percentage:.1f}%")

st.dataframe(high_risk)

st.markdown("---")

# ==============================
# 🔮 PREDICCIÓN
# ==============================

st.subheader("🔮 Simulación de entrega")

distance = st.slider("Distancia (km)", 1.0, 20.0)
traffic = st.selectbox("Tráfico", ["low", "medium", "high"])
weather = st.selectbox("Clima", ["clear", "rain", "storm"])

input_data = pd.DataFrame([{
    "distance": distance,
    "traffic": traffic,
    "weather": weather
}])

input_data = pd.get_dummies(input_data)

# 🔥 FIX PRO
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[features]

if st.button("Predecir ETA"):
    prediction = model.predict(input_data)[0]

    st.write(f"⏱️ Tiempo estimado: {prediction:.2f} minutos")

    if prediction > 40:
        st.error("⚠️ Alto riesgo de retraso")
    else:
        st.success("✅ Entrega en tiempo esperado")

    # 🔥 WOW MOMENT
    if traffic == "high" and weather != "clear":
        st.error("🚨 Escenario crítico: alta probabilidad de retraso")

st.markdown("---")

# ==============================
# 🧠 INSIGHTS
# ==============================

st.subheader("🧠 Insights clave")

st.write("""
- El tráfico alto es el principal factor de retraso  
- El clima adverso incrementa los tiempos  
- La distancia tiene menor impacto relativo  
- Los peores escenarios ocurren cuando hay tráfico alto + mal clima  
""")

# ==============================
# 📌 RECOMENDACIONES (NIVEL EMPRESA)
# ==============================

st.subheader("📌 Recomendaciones")

st.write("""
- Aumentar repartidores en zonas con tráfico alto  
- Ajustar tiempos estimados en días de lluvia  
- Priorizar pedidos críticos para evitar cancelaciones  
""")

# ==============================
# 💼 CONCLUSIÓN
# ==============================

st.subheader("💼 Conclusión")

st.write("""
El principal factor que afecta los tiempos de entrega es el tráfico, seguido por el clima.
Optimizar estos factores puede mejorar significativamente la eficiencia operativa.
""")