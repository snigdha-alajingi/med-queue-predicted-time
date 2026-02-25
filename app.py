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
# --- START: Random Forest & XGBoost Prediction ---
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv("hospital_wait_times.csv")  # your CSV file

# Features and target
X = data[["Queue Length", "Staff Available", "Emergency Level"]]  # input features
y = data["Estimated Waiting Time"]  # target column

# Split dataset for training (behind the scenes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Take user input (replace these with your actual Streamlit input variables)
queue_length = st.number_input("Queue Length", min_value=0, value=43)
staff_available = st.number_input("Staff Available", min_value=0, value=33)
emergency_level = st.number_input("Emergency Level", min_value=1, max_value=5, value=3)

user_input = np.array([[queue_length, staff_available, emergency_level]])

# Let user choose model
model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])

if model_choice == "Random Forest":
    prediction = rf_model.predict(user_input)[0]
else:
    prediction = xgb_model.predict(user_input)[0]

st.write("Predicted Waiting Time:", round(prediction, 1), "minutes")
# --- END: Random Forest & XGBoost Prediction ---

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

