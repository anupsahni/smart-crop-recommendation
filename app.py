import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="🌾 Smart Crop App", layout="wide")

# Dark Theme CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
.title {
    text-align: center;
    font-size: 45px;
    color: #00ffcc;
    font-weight: bold;
}
.sub {
    text-align: center;
    font-size: 18px;
    color: #ccc;
}
.result {
    text-align: center;
    font-size: 30px;
    color: #00ffcc;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🌱 Smart Crop Recommendation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">AI Powered Farming Assistant 🚀</p>', unsafe_allow_html=True)

# Load data
data = pd.read_csv("Crop_recommendation.csv")

X = data.drop("label", axis=1)
y = data["label"]

model = RandomForestClassifier()
model.fit(X, y)

# Layout
col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 0, 140, 50)
    K = st.slider("Potassium (K)", 0, 140, 50)

with col2:
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.slider("pH Value", 0.0, 14.0, 6.5)
    rain = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# Prediction
if st.button("🚀 Predict Crop"):
    result = model.predict([[N, P, K, temp, humidity, ph, rain]])

    st.markdown(f'<p class="result">🌾 Recommended Crop: {result[0]}</p>', unsafe_allow_html=True)

    # Chart
    st.subheader("📊 Soil Data Visualization")

    labels = ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rain']
    values = [N, P, K, temp, humidity, ph, rain]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🔥 Developed by You | AI + ML Project")