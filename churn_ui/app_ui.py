import streamlit as st
import requests
import json

# --- Page Config ---
st.set_page_config(
    page_title="Churn Prediction AI",
    page_icon="🔮",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("Telco Customer Churn")
st.markdown("Dashboard to predict if a customer is at risk of leaving. Adjust the customer profile below.")
st.write("---")

# --- Input Form ---
# We use columns to organize the "drop downs" cleanly
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.write("### 💳 Billing Details")
col4, col5, col6 = st.columns(3)

with col4:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col5:
    payment = st.selectbox("Payment Method", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])

with col6:
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=70.0, step=10.0)

# --- Logic to Send Data to API ---
if st.button("Predict Churn Risk"):
    
    # 1. Prepare the Data 
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    # 2. Send to AWS API
    api_url = "http://51.20.53.48:8000/predict"  
    
    try:
        with st.spinner("Analyzing customer data..."):
            response = requests.post(api_url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            
            # 3. Display Results
            st.success("Analysis Complete!")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Prediction", result['churn_prediction'])
            
            with res_col2:
                # Format probability as percentage
                prob = result['churn_probability'] * 100
                st.metric("Probability", f"{prob:.1f}%")
            
            with res_col3:
                risk = result['risk_level']
                color = "red" if risk == "Critical" else "green"
                st.markdown(f"**Risk Level:** :{color}[{risk}]")
                
            st.write("Churn Probability Score:")
            st.progress(result['churn_probability'])
            
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        st.error(f"Connection Failed. Is the AWS server running? Error: {e}")