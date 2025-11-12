# ================================
# 📘 EMI Prediction Streamlit App
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# 1️⃣  Page Configuration
# ----------------------------
st.set_page_config(
    page_title="EMI Prediction App",
    page_icon="💰",
    layout="centered"
)

st.title("💳 EMI Eligibility & Prediction System")
st.markdown("### Predict customer EMI eligibility and maximum EMI amount using ML models.")

# ----------------------------
# 2️⃣  Load Models and Scalers
# ----------------------------
@st.cache_resource
def load_models():
    try:
        classification_model = joblib.load("../models/classification_model.pkl")
        regression_model = joblib.load("../models/regression_model.pkl")
        scaler_class = joblib.load("../models/scaler.pkl")
        scaler_reg = joblib.load("../models/reg_scaler.pkl")
        return classification_model, regression_model, scaler_class, scaler_reg
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None, None

clf_model, reg_model, scaler_clf, scaler_reg = load_models()

if clf_model is None:
    st.stop()

# ----------------------------
# 3️⃣  Sidebar for Inputs
# ----------------------------
st.sidebar.header("📋 Enter Customer Details")

age = st.sidebar.number_input("Age", 18, 70, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", 1000, 1000000, 250000, step=10000)
loan_tenure = st.sidebar.slider("Loan Tenure (months)", 6, 60, 24)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
existing_loans = st.sidebar.number_input("Number of Existing Loans", 0, 10, 1)
employment_years = st.sidebar.slider("Years in Current Job", 0, 40, 5)

# Create DataFrame from inputs
input_data = pd.DataFrame({
    "Age": [age],
    "Monthly_Income": [income],
    "Loan_Amount": [loan_amount],
    "Loan_Tenure": [loan_tenure],
    "Credit_Score": [credit_score],
    "Existing_Loans": [existing_loans],
    "Employment_Years": [employment_years]
})

st.write("### 🔍 Input Data Preview")
st.dataframe(input_data)

# ----------------------------
# 4️⃣  Prediction Buttons
# ----------------------------

tab1, tab2 = st.tabs(["🏦 EMI Eligibility", "💰 Maximum EMI Prediction"])

with tab1:
    st.subheader("🏦 Check EMI Eligibility")
    if st.button("Check Eligibility"):
        try:
            X_scaled = scaler_clf.transform(input_data)
            pred = clf_model.predict(X_scaled)
            result = "✅ Eligible" if pred[0] == 1 else "❌ Not Eligible"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

with tab2:
    st.subheader("💰 Predict Maximum EMI Amount")
    if st.button("Predict EMI Amount"):
        try:
            X_scaled = scaler_reg.transform(input_data)
            emi_pred = reg_model.predict(X_scaled)[0]
            st.success(f"Predicted Maximum EMI: ₹ {emi_pred:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ----------------------------
# 5️⃣  Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by **Jabez Bodkhe** | Internship ML Project | EMI Prediction System 💻")
