#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
feature_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes',
    'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes',
    'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year',
    'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

df_knn_base_array = pickle.load(open("df_knn_base.pkl", "rb"))


# Now try to build the DataFrame
try:
    df_knn_base = pd.DataFrame(df_knn_base_array, columns=feature_columns)
except Exception as e:
    st.error(f"❌ Error converting df_knn_base_array to DataFrame: {e}")
    st.stop()
# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
kmeans_model = pickle.load(open("kmeans.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))

# Feature columns (must match training set)
feature_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes',
    'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes',
    'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year',
    'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]
# --- ML-based Recommendation Function ---
def generate_recommendation(user_input_scaled, kmeans_model,input_df):
    cluster = kmeans_model.predict(user_input_scaled)[0]

    recommendations = {
        0: ["Offer a 12-month discount plan.", "Provide free tech support."],
        1: ["Promote auto-payment methods with cashback.", "Bundle services for better value."],
        2: ["Send loyalty rewards.", "Suggest upgrades with minimal price change."],
        3: ["Engage with special offers via email.", "Assign a personal service rep."],
        4: ["Offer limited-time streaming deals.", "Provide proactive service checks."],
    }

    return "\n".join(recommendations.get(cluster, ["No specific recommendation found."]))

st.title("📉 Customer Churn Prediction App")
st.markdown("Provide customer details to predict churn and get churn prevention suggestions.")
def knn_recommendation(user_scaled, knn_model, df_knn_base):
    distances, indices = knn_model.kneighbors(user_scaled)
    similar_customers = df_knn_base.iloc[indices[0]]

    # Find most common contract type
    common_contract = similar_customers[['Contract_One year', 'Contract_Two year']].sum().idxmax()
    contract_suggestion = "Offer a discount for a longer contract." if common_contract == "Contract_One year" else "Provide perks for short-term contracts."

    # Find most common payment method
    common_payment = similar_customers[['PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']].sum().idxmax()
    payment_suggestion = {
        'PaymentMethod_Credit card (automatic)': "Encourage auto-pay with discounts.",
        'PaymentMethod_Electronic check': "Suggest secure online payments for better convenience.",
        'PaymentMethod_Mailed check': "Promote digital payments for ease."
    }.get(common_payment, "Consider flexible payment options.")

    # Find tech support preference
    tech_support_suggestion = "Provide free tech support." if similar_customers['TechSupport_Yes'].mean() < 0.5 else "Highlight premium tech support benefits."

    return " | ".join([contract_suggestion, payment_suggestion, tech_support_suggestion])

# ---- User Input Section ---- #
gender = st.selectbox("Gender", ['Female', 'Male'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ['No', 'Yes'])
Dependents = st.selectbox("Dependents", ['No', 'Yes'])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ['No', 'Yes'])
MultipleLines = st.selectbox("Multiple Lines", ['No', 'Yes'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes'])
OnlineBackup = st.selectbox("Online Backup", ['No', 'Yes'])
DeviceProtection = st.selectbox("Device Protection", ['No', 'Yes'])
TechSupport = st.selectbox("Tech Support", ['No', 'Yes'])
StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes'])
StreamingMovies = st.selectbox("Streaming Movies", ['No', 'Yes'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['No', 'Yes'])
PaymentMethod = st.selectbox("Payment Method", ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# ---- Encode Input ---- #
input_dict = {
    'SeniorCitizen': SeniorCitizen,
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'gender_Male': 1 if gender == 'Male' else 0,
    'Partner_Yes': 1 if Partner == 'Yes' else 0,
    'Dependents_Yes': 1 if Dependents == 'Yes' else 0,
    'PhoneService_Yes': 1 if PhoneService == 'Yes' else 0,
    'MultipleLines_Yes': 1 if MultipleLines == 'Yes' else 0,
    'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
    'InternetService_No': 1 if InternetService == 'No' else 0,
    'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
    'OnlineBackup_Yes': 1 if OnlineBackup == 'Yes' else 0,
    'DeviceProtection_Yes': 1 if DeviceProtection == 'Yes' else 0,
    'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
    'StreamingTV_Yes': 1 if StreamingTV == 'Yes' else 0,
    'StreamingMovies_Yes': 1 if StreamingMovies == 'Yes' else 0,
    'Contract_One year': 1 if Contract == 'One year' else 0,
    'Contract_Two year': 1 if Contract == 'Two year' else 0,
    'PaperlessBilling_Yes': 1 if PaperlessBilling == 'Yes' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0
}

# Create DataFrame and align
input_df = pd.DataFrame([input_dict])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# Scale
input_scaled = scaler.transform(input_df)
# Toggle for selecting recommendation method
reco_method = st.radio(
    "Choose Recommendation Method for Churn Prevention:",
    ("KMeans : generalized", "KNN:personalized", "Both")
)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    if prediction== 1:
        st.error(f"⚠️ Predicted: Churn (Probability: {prediction_proba:.2f})")
        if reco_method in ["KMeans : generalized", "Both"]:
            st.markdown("### 💡 KMeans-Based Recommendations:")
            st.info(generate_recommendation(input_scaled, kmeans_model, input_df))

        if reco_method in ["KNN:personalized", "Both"]:
            st.markdown("### 💡 KNN-Based Recommendations:")
            st.info(knn_recommendation(input_scaled, knn_model, df_knn_base))
       
    else:
        st.success(f"✅ Predicted: No Churn (Probability: {prediction_proba:.2f})")

       
    





