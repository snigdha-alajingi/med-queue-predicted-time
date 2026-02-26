import streamlit as st
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from xgboost import XGBRegressor

# ===============================
# Google Sheets Setup
# ===============================
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Sheets
sheet_hospitals = client.open("HospitalQueueData").worksheet("Hospitals")
sheet_appointments = client.open("HospitalQueueData").worksheet("Appointments")
sheet_logs = client.open("HospitalQueueData").worksheet("Logs")

# ===============================
# Load Hospital Data
# ===============================
hospital_df = pd.DataFrame(sheet_hospitals.get_all_records())
hospital_df['Hospital'] = hospital_df['Hospital'].str.strip()
hospital_df['City'] = hospital_df['City'].str.strip()

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="🏥 Global Med-Queue AI", layout="wide")
st.title("🏥 Global Med-Queue AI")

# --- Select City/Town/Village/Country ---
cities = sorted(hospital_df['City'].unique())
selected_city = st.selectbox("Select City / Town / Village / Country:", cities)

city_df = hospital_df[hospital_df['City'] == selected_city]
hospital_list = city_df['Hospital'].tolist()

if hospital_list:
    selected_hospital = st.selectbox("Select Hospital:", hospital_list)

    # Fetch current live values from sheet
    hosp_row = hospital_df[hospital_df['Hospital'] == selected_hospital].iloc[0]
    queue_length = int(hosp_row['Queue'])
    staff_available = int(hosp_row['Staff'])
    emergency_level = int(hosp_row['Emergency'])
    hospital_website = hosp_row.get('Website', "https://example.com")

    # ===============================
    # Train ML Wait Time Model
    # ===============================
    np.random.seed(0)
    X_synth = np.column_stack([
        np.random.randint(5, 60, 1000),
        np.random.randint(5, 50, 1000),
        np.random.randint(1, 4, 1000)
    ])
    y_synth = (X_synth[:,0] * 2 / np.maximum(X_synth[:,1],1) * X_synth[:,2] * 10).astype(int)

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_synth, y_synth)

    # ===============================
    # Patient Booking Form
    # ===============================
    st.markdown("### 📝 Book Appointment")
    with st.form(key="appt_form"):
        pname = st.text_input("Full Name")
        page = st.number_input("Age", min_value=0, max_value=120, value=30)
        pgender = st.selectbox("Gender", ["Male", "Female", "Other"])
        pcontact = st.text_input("Contact Number")
        pemerg = st.selectbox("Emergency Level", ["Low", "Medium", "High"])
        submit = st.form_submit_button("Submit")

    if submit:
        # Map
        level_map = {"Low":1,"Medium":2,"High":3}
        pnum = level_map[pemerg]

        queue_length += 1
        staff_available = max(1, staff_available + np.random.randint(-2,3))
        emergency_level = np.random.choice([1,2,3])

        # Predict wait
        arr = np.array([[queue_length, staff_available, pnum]])
        est_wait = int(round(xgb.predict(arr)[0],0))

        # Update sheet
        cell = sheet_hospitals.find(selected_hospital)
        sheet_hospitals.update_cell(cell.row, 3, queue_length)  # Queue
        sheet_hospitals.update_cell(cell.row, 4, staff_available)
        sheet_hospitals.update_cell(cell.row, 5, emergency_level)

        sheet_appointments.append_row([
            selected_hospital, pname, page, pgender, pcontact, pemerg, est_wait
        ])

        sheet_logs.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "New Booking",
            selected_hospital,
            f"{pname}, {pemerg}, wait={est_wait}"
        ])

        st.success(f"✅ Appointment booked! Est. wait: {est_wait} mins")

    # ===============================
    # Display Live Hospital Status
    # ===============================
    st.markdown("### 📊 Live Hospital Status")
    st.write(f"🧍 Queue Length: {queue_length}")
    st.write(f"👨‍⚕ Staff Available: {staff_available}")
    em_text = {1:"Low",2:"Medium",3:"High"}.get(emergency_level,"Unknown")
    st.write(f"🚨 Emergency Level: {em_text}")
    st.markdown(f"🌐 Hospital Website: [{hospital_website}]({hospital_website})")

    # ===============================
    # Display Upcoming Appointments
    # ===============================
    appt_df = pd.DataFrame(sheet_appointments.get_all_records())
    appt_df = appt_df[appt_df['Hospital'] == selected_hospital]
    if not appt_df.empty:
        st.markdown("### 📋 Upcoming Appointments")
        st.dataframe(appt_df)

else:
    st.warning("⚠ No hospitals found here. Try another city.")
    
