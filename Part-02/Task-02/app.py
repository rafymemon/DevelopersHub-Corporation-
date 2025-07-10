import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("telco_churn_model.joblib")

# App title
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Predict whether a customer will churn based on their service usage and contract information.")

# Sidebar for user input
st.sidebar.header("Enter Customer Details")

# User inputs
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=3000.0)

    # Create DataFrame for prediction
    data = {
        "Gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure": [tenure],  # Changed from 'tenure'
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "ContractType": [contract],  # Changed from 'Contract'
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Age": [senior * 65]  # Just for placeholder if you added 'Age' during training
    }

    return pd.DataFrame(data)

input_df = user_input_features()

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn (probability: {proba:.2f})")
    else:
        st.success(f"✅ This customer is unlikely to churn (probability: {proba:.2f})")

# Show user input for review
st.subheader("🔎 Input Summary")
st.write(input_df)
