import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

with open('data.pkl', 'rb') as t:
    df_train = pickle.load(t)
# Load the pre-trained model
with open('model_binary.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Load the data model to extract levels (assuming `df_train` is a pandas DataFrame)
# You may need to load it or define it according to your dataset

# Initialize label encoders for categorical variables
paypoint_name_le = LabelEncoder()
paypoint_name_le.fit(df_train['PAYPOINT_NAME'])

payment_mode_le = LabelEncoder()
payment_mode_le.fit(df_train['PAYMENT_MODE'])

insured_id_desc_le = LabelEncoder()
insured_id_desc_le.fit(df_train['INSURED_ID_DESCRIPTION'])

company_name_le = LabelEncoder()
company_name_le.fit(df_train['COMPANY_NAME'])

marital_status_le = LabelEncoder()
marital_status_le.fit(df_train['MARITAL_STATUS'])

product_code_le = LabelEncoder()
product_code_le.fit(df_train['PRODUCT_CODE'])

# Streamlit UI
st.title("Policy Status Prediction App: Binary")

paypoint_name = st.selectbox("Paypoint Name:", options=df_train['PAYPOINT_NAME'].unique())
premium = st.number_input("Premium:", min_value=0, value=300)
income = st.number_input("Income:", min_value=0, value=7000)
payment_mode = st.selectbox("Payment Mode:", options=df_train['PAYMENT_MODE'].unique())
insured_id_description = st.selectbox("Insured ID Type:", options=[1, 2])
term = st.number_input("Term:", min_value=1, value=60)
company_name = st.selectbox("Company Name:", options=df_train['COMPANY_NAME'].unique())
marital_status = st.selectbox("Marital Status:", options=df_train['MARITAL_STATUS'].unique())
product_code = st.selectbox("Product Code:", options=df_train['PRODUCT_CODE'].unique())

predict_btn = st.button("Predict Status")

if predict_btn:
    # Create a new client data
    new_client = pd.DataFrame({
        'PAYPOINT_NAME': [paypoint_name_le.transform([paypoint_name])[0]],
        'PREMIUM': [np.log1p(premium)],
        'INCOME': [np.log1p(income)],
        'PAYMENT_MODE': [payment_mode_le.transform([payment_mode])[0]],
        'INSURED_ID_DESCRIPTION': [insured_id_desc_le.transform([insured_id_description])[0]],
        'TERM': [term],
        'COMPANY_NAME': [company_name_le.transform([company_name])[0]],
        'MARITAL_STATUS': [marital_status_le.transform([marital_status])[0]],
        'PRODUCT_CODE': [product_code_le.transform([product_code])[0]]
    })
    
    # Prepare input for prediction (similar to `model.matrix` in R)
    new_client_ = new_client.values  # Ensure it's in the correct format for your model
    
    # Predict status
    predicted_status = model.predict(new_client_)
    def get_status(predicted_status):
        if predicted_status == 0:
            return "Active"
        elif predicted_status == 1:
            return "Lapse"
        else:
            return 'Unknown'
    predicted_status = get_status(predicted_status)
    # Display prediction result
    predicted_proba = pd.DataFrame(model.predict_proba(new_client_))
    predicted_proba.rename(columns={0: 'Active Class', 1: 'Lapsed Class'},index={0: 'Predicted Probability'},inplace=True)
    # Display prediction result
    st.subheader(f"Predicted Status: {predicted_status}")
    st.table(predicted_proba)
