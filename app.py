import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="centered")
st.title("📞 Telecom Customer Churn Risk Predictor")

@st.cache_resource
def load_model():
    try:
        return joblib.load("churn_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file 'churn_model.pkl' not found. Please train and save your pipeline first!")
        return None

pipeline = load_model()

if pipeline is not None:
    
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox("Region", ["Midwest", "Northeast", "South", "West"]) 
        vmail_messages = st.slider("Number of Voicemail messages")
        total_day_charge = st.number_input("Total Day Charge ($)")
        total_eve_charge = st.number_input("Total Evening Charge ($)")
    
    with col2:
        total_intl_calls = st.slider("Number of international calls")
        total_intl_charge = st.number_input("Total International Charge ($)")
        customer_service_calls = st.slider("Number of Customer Service Calls")
        international_plan = st.selectbox("International Plan", ["No", "Yes"])

    input_data = pd.DataFrame({
        "Region": [region],
        "Number vmail messages": [vmail_messages],
        "Total day charge": [total_day_charge],
        "Total eve charge": [total_eve_charge],
        "Total intl calls": [total_intl_calls],
        "Total intl charge": [total_intl_charge],
        "Customer service calls": [customer_service_calls],
        "International plan": [international_plan]

    })

    st.markdown("---")

    if st.button("🚀 Evaluate Churn Risk", type="primary"):
        probability = pipeline.predict_proba(input_data)[0][1]
        
        st.subheader("Prediction Result")
        
        if probability >= 0.5:
            st.error(f"🚨 **High Risk of Churning!** (Probability: {probability:.2%})")
        else:
            st.success(f"✅ **Low Risk of Churning** (Probability: {probability:.2%})")
           
