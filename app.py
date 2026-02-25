# app.py
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ===============================
# Load Data
# ===============================
df = pd.read_csv("HospitalsInIndia.csv")
df.columns = df.columns.str.strip()
df['City'] = df['City'].str.strip()
df['Hospital'] = df['Hospital'].str.strip()

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="🏥 Global Med-Queue AI", layout="wide")
st.title("🏥 Global Med-Queue AI")

# --- Select City/Town/Village/Country ---
unique_cities = sorted(df['City'].unique())
selected_city = st.selectbox("Select City / Town / Village / Country:", unique_cities)

# Filter hospitals in selected city
filtered_hospitals = df[df['City'] == selected_city]
hospital_names = filtered_hospitals['Hospital'].tolist()

if hospital_names:
    selected_hospital = st.selectbox("Select Hospital:", hospital_names)

    # ===============================
    # Generate demo operational data
    # ===============================
    np.random.seed(42)
    queue_length = np.random.randint(5, 60)
    staff_available = np.random.randint(5, 50)
    emergency_level = np.random.choice([1, 2, 3])  # 1=Low, 2=Medium, 3=High

    # ===============================
    # Synthetic ML training data
    # ===============================
    np.random.seed(0)
    X_synthetic = np.column_stack([
        np.random.randint(5, 60, 1000),   # Queue Length
        np.random.randint(5, 50, 1000),   # Staff Available
        np.random.randint(1, 4, 1000)     # Emergency Level
    ])
    y_synthetic = (X_synthetic[:,0] * 2 / np.maximum(X_synthetic[:,1],1) * X_synthetic[:,2] * 10).astype(int)

    # Train XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_synthetic, y_synthetic)

    # Predict waiting time
    user_input = np.array([[queue_length, staff_available, emergency_level]])
    wait_time = int(round(xgb_model.predict(user_input)[0], 0))

    # ===============================
    # Display Results
    # ===============================
    st.markdown("### 📊 Live Hospital Status")
    st.write(f"🧍 Queue Length: {queue_length}")
    st.write(f"👨‍⚕ Staff Available: {staff_available}")
    emergency_text = {1: "Low", 2: "Medium", 3: "High"}[emergency_level]
    st.write(f"🚨 Emergency Level: {emergency_text}")
    st.write(f"⏳ Estimated Waiting Time: {wait_time} minutes")

    st.markdown("### 🏥 Instructions & Advice")
    if emergency_level == 3:
        st.write("⚠ Hospital is crowded, expect longer wait times.")
    else:
        st.write("Low crowd. Faster service expected.")

    st.write("✔ Carry valid ID")
    st.write("✔ Bring previous medical reports")
    st.write("✔ Arrive 15 minutes early")

    # Optional map if latitude/longitude exist
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.map(filtered_hospitals[['Latitude', 'Longitude']])

else:
    st.warning("⚠ No hospitals found in this location. Try another city/town/village.")
