# 🚀 OpsVision AI — Delivery Operations Intelligence

## 📦 Overview
OpsVision AI es un sistema de análisis de operaciones que permite entender por qué se retrasan las entregas y predecir tiempos de entrega (ETA) usando Machine Learning.

Incluye:
- Dashboard interactivo
- Modelo predictivo
- Simulación de escenarios

🔗 Live Demo: https://opsvision-ai.streamlit.app/

---

## 🎯 Problema de negocio

Las empresas de delivery enfrentan:
- Retrasos constantes
- Mala estimación de tiempos
- Baja satisfacción del cliente

💰 Impacto:
- Cancelaciones
- Costos operativos altos
- Mala experiencia de usuario

---

## 🧠 Solución

Se desarrolló un sistema que:

- Analiza factores como tráfico, clima y distancia  
- Identifica causas de retrasos  
- Predice tiempos de entrega  
- Permite simular escenarios en tiempo real  

---

## 📊 Dataset

Variables:
- distance → distancia del pedido  
- traffic → tráfico (low, medium, high)  
- weather → clima  
- delivery_time → tiempo de entrega  

---

## 🛠️ Stack

- Python (pandas, numpy)
- scikit-learn
- Streamlit
- Matplotlib
- Joblib

---

## 📈 Insights clave

- El tráfico es el principal factor de retraso  
- El clima empeora significativamente los tiempos  
- La distancia tiene menor impacto relativo  

---

## 📌 Recomendaciones

- Aumentar repartidores en tráfico alto  
- Ajustar tiempos en condiciones climáticas adversas  
- Priorizar pedidos críticos  

---

## 🖥️ Cómo ejecutar

```bash
pip install -r requirements.txt
python src/train.py
python -m streamlit run app/dashboard.py