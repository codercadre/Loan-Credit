import streamlit as st
import pandas as pd
import joblib

# Load the saved models and reference data
linear_model = joblib.load('linear_model.pkl')
logistic_model_scaled = joblib.load('logistic_model_scaled.pkl')
scaler = joblib.load('scaler.pkl')
reference_row = joblib.load('reference_row.pkl')

st.title("ABC Ltd Credit Prediction Tool")
st.markdown("Assess optimal interest rates and default probabilities for new loan applications.")

# Sidebar inputs for the Manager
st.sidebar.header("Applicant Profile")
loan_amnt = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000, value=15000)
term = st.sidebar.selectbox("Loan Term (Months)", [36, 60])
annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=10000, value=75000)
dti = st.sidebar.slider("Debt-to-Income Ratio (DTI %)", 0.0, 50.0, 15.0)

if st.button("Generate AI Prediction"):
    # Copy the reference row to maintain the 78-column structure
    input_df = reference_row.copy()
    
    # Update with the specific inputs from the sidebar
    input_df['loan_amnt'] = loan_amnt
    input_df['term'] = term
    input_df['annual_inc'] = annual_inc
    input_df['dti'] = dti
    
    # 1. Linear Prediction (Interest Rate)
    predicted_rate = linear_model.predict(input_df)[0]
    
    # 2. Logistic Prediction (Default Risk)
    input_scaled = scaler.transform(input_df)
    # predict_proba returns [prob_0, prob_1]. Class 0 was mapped to 'Charged Off'
    default_prob = logistic_model_scaled.predict_proba(input_scaled)[0][0] * 100 
    
    # Display the results cleanly
    col1, col2 = st.columns(2)
    col1.metric("Predicted Optimal Interest Rate", f"{predicted_rate:.2f}%")
    col2.metric("Predicted Risk of Default", f"{default_prob:.1f}%")
    
    st.markdown("---")
    if default_prob > 30:
        st.error("⚠️ **High Risk:** The model indicates significant default probability. Manual underwriter review is strongly recommended.")
    else:
        st.success("✅ **Acceptable Risk:** Applicant matches typical profile for expedited approval workflow.")
