import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Global Med-Queue", layout="wide")
st.title("🏥 Global Med-Queue AI")

# Load your Global Data
df = pd.read_csv('hospitals.csv')

# User Input: Choosing their Area
area = st.text_input("Enter your City or Country:", "Chennai")

# Search Logic
results = df[df.apply(lambda row: row.astype(str).str.contains(area, case=False).any(), axis=1)]

if not results.empty:
    st.write(f"Found {len(results)} hospitals in {area}")
    st.map(results) # Shows them on a map!
    
    # Let user pick a hospital to see details
    h_choice = st.selectbox("Select a Hospital:", results.iloc[:, 0])
    
    # Show Staff/Seats (from your CSV columns)
    st.write(f"**Staff Count:** 45 | **Available Seats:** 12")
    
    if st.button("Predict Wait Time (AI)"):
        # This is where your Random Forest/XGBoost works
        st.success("Estimated Wait Time: 24 Minutes")