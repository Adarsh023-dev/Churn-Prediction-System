import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("../Models/churn_model.pkl", "rb"))

st.title("📊 Customer Churn Prediction System")

st.write("Enter customer details to predict churn risk")

# INPUTS
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
tenure = st.number_input("Tenure (months)", 0, 100, 10)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# SIMPLE ENCODING (must match training encoding)
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

# PREDICTION BUTTON
if st.button("Predict Churn"):
    
    input_data = np.array([[monthly_charges, tenure, contract_map[contract]]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer will STAY")