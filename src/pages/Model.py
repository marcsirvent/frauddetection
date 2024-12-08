import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model
@st.cache_data()
def load_model():
    with open('./data/final_model.pkl', 'rb') as file:
        model_dict = pickle.load(file)
    model = model_dict.get('model')
    if model is None:
        st.error("No model found in the provided file.")
        st.stop()
    return model

model = load_model()

st.title("Fraud Detection Prediction")

st.markdown("""
**Enter the details of a transaction to predict whether it might be fraudulent.**  
Some fields are technical or hard to know. If unsure, leave default values or select from given options.
""")

# For attributes the user is likely to know or can reasonably guess:
transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=33.06, step=0.5)
transaction_hour = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
transaction_month = st.selectbox("Transaction Month", 
                                 options=["January","February","March","April","May","June","July","August","September","October","November","December"], 
                                 index=6)  
transaction_dayofweek = st.selectbox("Day of Week", 
                                     options=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], 
                                     index=3)  
use_chip_online = st.radio("Was it an Online Transaction?", ["No", "Yes"])
use_chip_online_val = 1 if use_chip_online == "Yes" else 0

# For attributes a user might not know:
st.markdown("**Advanced/Technical Fields (If unsure, use defaults):**")
mcc = st.number_input("MCC (Merchant Category Code)", min_value=0, value=5499, help="If unknown, leave default")
merchant_id = st.number_input("Merchant ID", min_value=0, value=47399, help="If unknown, leave default")
longitude = st.number_input("Transaction Longitude", value=-86.52, help="If unknown, leave default")
credit_limit = st.number_input("Credit Limit", min_value=0, value=13322, help="If unknown, leave default")
transaction_year = st.selectbox("Transaction Year", [2015, 2016, 2017, 2018, 2019], index=0)

# Convert categorical inputs to numeric
month_map = {
    "January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
    "July":7, "August":8, "September":9, "October":10, "November":11, "December":12
}
dow_map = {
    "Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3,
    "Friday":4, "Saturday":5, "Sunday":6
}

input_data = pd.DataFrame([{
    'mcc': mcc,
    'transaction_year': transaction_year,
    'transaction_hour': transaction_hour,
    'transaction_month': month_map[transaction_month],
    'credit_limit': credit_limit,
    'merchant_id': merchant_id,
    'transaction_amount': transaction_amount,
    'transaction_dayofweek': dow_map[transaction_dayofweek],
    'use_chip_Online Transaction': use_chip_online_val,
    'longitude': longitude
}])

if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1] * 100
    
    if prediction == 1:
        st.markdown("### The transaction is predicted to be FRAUDULENT.")
    else:
        st.markdown("### The transaction is predicted to be NOT FRAUDULENT.")

    st.write(f"**Probability of Fraud:** {prediction_proba:.4f} %")
else:
    st.write("Click **Predict Fraud** to see the model's prediction.")