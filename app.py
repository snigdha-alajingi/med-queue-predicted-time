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
if 'Website' not in df.columns:
    df['Website'] = "https://example.com"

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
    # Session state for dynamic hospital data
    # ===============================
    if 'hospital_data' not in st.session_state:
        st.session_state.hospital_data = {}
    if selected_hospital not in st.session_state.hospital_data:
        np.random.seed(hash(selected_hospital) % 2**32)
        st.session_state.hospital_data[selected_hospital] = {
            "queue_length": np.random.randint(5, 60),
            "staff_available": np.random.randint(5, 50),
            "emergency_level": np.random.choice([1, 2, 3]),
            "appointments": []  # List to store patient bookings
        }

    hospital_status = st.session_state.hospital_data[selected_hospital]

    # ===============================
    # Synthetic ML training data
    # ===============================
    np.random.seed(0)
    X_synthetic = np.column_stack([
        np.random.randint(5, 60, 1000),
        np.random.randint(5, 50, 1000),
        np.random.randint(1, 4, 1000)
    ])
    y_synthetic = (X_synthetic[:,0] * 2 / np.maximum(X_synthetic[:,1],1) * X_synthetic[:,2] * 10).astype(int)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_synthetic, y_synthetic)

    # ===============================
    # Patient pre-booking form
    # ===============================
    st.markdown("### 📝 Patient Pre-Booking Form")
    with st.form(key='booking_form'):
        patient_name = st.text_input("Full Name")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        patient_contact = st.text_input("Contact Number")
        patient_emergency_level = st.selectbox(
            "Emergency Level", ["Low", "Medium", "High"], index=0
        )
        submit_button = st.form_submit_button(label="Submit Appointment Request")

    if submit_button:
        # Map emergency level text to numeric
        emergency_level_map = {"Low": 1, "Medium": 2, "High": 3}
        patient_emergency_numeric = emergency_level_map[patient_emergency_level]

        # Update hospital queue dynamically
        hospital_status['queue_length'] += 1
        hospital_status['staff_available'] = max(1, hospital_status['staff_available'] + np.random.randint(-2, 3))
        hospital_status['emergency_level'] = np.random.choice([1, 2, 3])

        # Predict waiting time
        user_input = np.array([[hospital_status['queue_length'],
                                hospital_status['staff_available'],
                                patient_emergency_numeric]])
        wait_time = int(round(xgb_model.predict(user_input)[0], 0))

        # Save appointment
        hospital_status['appointments'].append({
            "Name": patient_name,
            "Age": patient_age,
            "Gender": patient_gender,
            "Contact": patient_contact,
            "Emergency": patient_emergency_level,
            "Estimated Wait (min)": wait_time
        })

        st.success("✅ Appointment request submitted successfully!")

    # ===============================
    # Display live hospital status
    # ===============================
    st.markdown("### 📊 Live Hospital Status")
    st.write(f"🧍 Queue Length: {hospital_status['queue_length']}")
    st.write(f"👨‍⚕ Staff Available: {hospital_status['staff_available']}")
    emergency_text = {1: "Low", 2: "Medium", 3: "High"}[hospital_status['emergency_level']]
    st.write(f"🚨 Emergency Level: {emergency_text}")

    # ===============================
    # Display upcoming appointments
    # ===============================
    if hospital_status['appointments']:
        st.markdown("### 📋 Upcoming Appointments")
        df_appointments = pd.DataFrame(hospital_status['appointments'])
        st.dataframe(df_appointments)

    # Hospital website
    website = filtered_hospitals[filtered_hospitals['Hospital'] == selected_hospital]['Website'].values[0]
    st.markdown(f"🌐 Visit Hospital Website: [{website}]({website})")

    # Optional map
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.map(filtered_hospitals[['Latitude', 'Longitude']])

else:
    st.warning("⚠ No hospitals found in this location. Try another city/town/village.")
