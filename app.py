import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('notebooks\model.pkl')
scaler = joblib.load('notebooks\scaler.pkl')  # Ensure you saved this from your notebook

# Feature names used for prediction
features = [
    'MedInc',        # Median income
    'HouseAge',      # Housing median age
    'AveRooms',      # Average rooms
    'AveBedrms',     # Average bedrooms
    'Population',    # Population
    'AveOccup',      # Average occupancy
    'Latitude',      # Latitude
    'Longitude'      # Longitude
]

# Streamlit UI
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")
st.title("🏡 California House Price Prediction App")

st.subheader("📋 Enter House Features")

# Input form for all features
input_data = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0, format="%.4f")
    input_data.append(val)

# Predict button
if st.button("Predict Price"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated House Price: ₹ {prediction * 1e7:,.2f}") 