# app.py
import streamlit as st
import pandas as pd
import numpy as np

# ===============================
# Load Data
# ===============================
df = pd.read_csv("HospitalsInIndia.csv")
df['City'] = df['City'].str.strip()
df['Hospital'] = df['Hospital'].str.strip()

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="🏥 Global Med-Queue AI", layout="wide")
st.title("🏥 Global Med-Queue AI")
st.write("Find hospitals and check estimated emergency wait time")

# --- Select City/Town/Village/Country ---
unique_cities = sorted(df['City'].unique())
selected_city = st.selectbox("Select City / Town / Village / Country:", unique_cities)

# Filter hospitals in selected city
filtered_hospitals = df[df['City'] == selected_city]
hospital_names = filtered_hospitals['Hospital'].tolist()

if hospital_names:
    selected_hospital = st.selectbox("Select Hospital:", hospital_names)

    # Generate synthetic operational data
    np.random.seed(42)
    queue_length = np.random.randint(5, 60)
    staff_available = np.random.randint(5, 50)
    emergency_level = np.random.choice(["Low", "Medium", "High"])

    # Simple formula for estimated wait time
    wait_time = int(queue_length * 2 / max(staff_available, 1) * 10)

    # Display Results
    st.markdown("### 📊 Live Hospital Status")
    st.write(f"🧍 Queue Length: {queue_length}")
    st.write(f"👨‍⚕ Staff Available: {staff_available}")
    st.write(f"🚨 Emergency Level: {emergency_level}")
    st.write(f"⏳ Estimated Waiting Time: {wait_time} minutes")

    st.markdown("### 🏥 Instructions & Advice")
    if emergency_level == "High":
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
