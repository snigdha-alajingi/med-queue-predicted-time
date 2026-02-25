# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ===============================
# Load Data
# ===============================
df = pd.read_csv("HospitalsInIndia.csv")

# Strip spaces from column names and text
df.columns = df.columns.str.strip()
df['City'] = df['City'].str.strip()
df['Hospital'] = df['Hospital'].str.strip()

# Check what your emergency column is called
st.write("Columns in CSV:", df.columns.tolist())

# Replace this with the exact column name from your CSV
emergency_col = "Emergency Level"  # <- CHANGE if your CSV uses a different name

# If emergency column is categorical, map it to numbers
emergency_mapping = {"Low": 1, "Medium": 2, "High": 3}
if df[emergency_col].dtype == object:
    df[emergency_col] = df[emergency_col].map(emergency_mapping)

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
    # ML Prediction Section
    # ===============================

    # Use the same CSV for ML features
    data = filtered_hospitals.copy()  # only hospitals in the selected city

    # Features and target (replace column names with exact ones from CSV)
    X = data[["Queue Length", "Staff Available", emergency_col]]
    y = data["Estimated Waiting Time"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Generate demo values (or you can take input from user if you want)
    queue_length = np.random.randint(5, 60)
    staff_available = np.random.randint(5, 50)
    emergency_level = np.random.choice([1, 2, 3])  # numerical

    user_input = np.array([[queue_length, staff_available, emergency_level]])

    # Choose model
    model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])
    if model_choice == "Random Forest":
        prediction = rf_model.predict(user_input)[0]
    else:
        prediction = xgb_model.predict(user_input)[0]

    # ===============================
    # Display Results
    # ===============================
    st.markdown("### 📊 Live Hospital Status")
    st.write(f"🧍 Queue Length: {queue_length}")
    st.write(f"👨‍⚕ Staff Available: {staff_available}")
    emergency_text = {1: "Low", 2: "Medium", 3: "High"}[emergency_level]
    st.write(f"🚨 Emergency Level: {emergency_text}")
    st.write(f"⏳ Estimated Waiting Time: {int(round(prediction, 0))} minutes")

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
