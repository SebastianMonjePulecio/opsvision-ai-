import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==============================
# 🎨 CONFIG UI
# ==============================

st.set_page_config(
    page_title="OpsVision AI",
    page_icon="🚀",
    layout="wide"
)

# ==============================
# 💅 CSS (SaaS LOOK)
# ==============================

st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    text-align: center;
}

h1, h2, h3 {
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# 📂 DATA
# ==============================

df = pd.read_csv("data/raw/deliveries.csv")
model = joblib.load("models/eta_model.pkl")
features = joblib.load("models/features.pkl")

# ==============================
# 🧠 HEADER
# ==============================

st.title("🚀 OpsVision AI")

st.markdown("""
### Delivery Operations Intelligence

Monitorea, analiza y predice tiempos de entrega en tiempo real.
""")

st.markdown("---")

# ==============================
# 🎛️ SIDEBAR
# ==============================

st.sidebar.title("⚙️ Filtros")

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
# 📊 KPIs (SaaS STYLE)
# ==============================

avg_time = filtered_df["delivery_time"].mean()
max_time = filtered_df["delivery_time"].max()
min_time = filtered_df["delivery_time"].min()

col1, col2, col3 = st.columns(3)

col1.metric("⏱️ Tiempo Promedio", f"{avg_time:.2f} min")
col2.metric("📈 Máximo", f"{max_time} min")
col3.metric("📉 Mínimo", f"{min_time} min")

st.markdown("---")

# ==============================
# 📊 HISTOGRAMA
# ==============================

st.subheader("📊 Distribución de entregas")

fig = px.histogram(
    filtered_df,
    x="delivery_time",
    nbins=20,
    title="Distribución de tiempos",
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# 🚗 TRÁFICO
# ==============================

st.subheader("🚗 Impacto del tráfico")

traffic_analysis = filtered_df.groupby("traffic")["delivery_time"].mean().reset_index()

fig = px.bar(
    traffic_analysis,
    x="traffic",
    y="delivery_time",
    color="traffic",
    title="Tiempo promedio por tráfico",
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# 🌧️ CLIMA
# ==============================

st.subheader("🌧️ Impacto del clima")

weather_analysis = filtered_df.groupby("weather")["delivery_time"].mean().reset_index()

fig = px.bar(
    weather_analysis,
    x="weather",
    y="delivery_time",
    color="weather",
    title="Tiempo promedio por clima",
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# 🔥 HEATMAP (WOW)
# ==============================

st.subheader("🔥 Tráfico + Clima")

pivot = filtered_df.pivot_table(
    values="delivery_time",
    index="traffic",
    columns="weather",
    aggfunc="mean"
)

fig = px.imshow(
    pivot,
    text_auto=True,
    aspect="auto",
    title="Impacto combinado"
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# ⚠️ RIESGO
# ==============================

st.subheader("⚠️ Riesgo de retraso")

high_risk = filtered_df[filtered_df["delivery_time"] > 40]

risk_percentage = (len(high_risk) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0

col1, col2 = st.columns(2)

col1.metric("Pedidos en riesgo", len(high_risk))
col2.metric("% en riesgo", f"{risk_percentage:.1f}%")

st.dataframe(high_risk)

# ==============================
# 🔮 PREDICCIÓN
# ==============================

st.markdown("---")
st.subheader("🔮 Simulación")

distance = st.slider("Distancia (km)", 1.0, 20.0)
traffic = st.selectbox("Tráfico", ["low", "medium", "high"])
weather = st.selectbox("Clima", ["clear", "rain", "storm"])

input_data = pd.DataFrame([{
    "distance": distance,
    "traffic": traffic,
    "weather": weather
}])

input_data = pd.get_dummies(input_data)

for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[features]

if st.button("Predecir ETA"):
    prediction = model.predict(input_data)[0]

    st.success(f"⏱️ ETA estimado: {prediction:.2f} minutos")

    if prediction > 40:
        st.error("⚠️ Riesgo alto")

    if traffic == "high" and weather != "clear":
        st.error("🚨 Escenario crítico")

# ==============================
# 🧠 INSIGHTS
# ==============================

st.markdown("---")
st.subheader("🧠 Insights")

st.write("""
- El tráfico impacta más que la distancia  
- El clima empeora los tiempos  
- Los peores casos combinan tráfico alto + lluvia  
""")

# ==============================
# 📌 RECOMENDACIONES
# ==============================

st.subheader("📌 Recomendaciones")

st.write("""
- Aumentar repartidores en tráfico alto  
- Ajustar tiempos en lluvia  
- Priorizar pedidos críticos  
""")

# ==============================
# 💼 CONCLUSIÓN
# ==============================

st.subheader("💼 Conclusión")

st.write("""
Optimizar tráfico y clima puede mejorar significativamente la eficiencia operativa y la experiencia del cliente.
""")