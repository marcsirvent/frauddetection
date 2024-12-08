import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

GLOBAL_PATH = './data/'
TRAIN_DATASET_SAMPLE = "train_df_sample.csv"
TRAIN_DATASET = "train_df.csv"

@st.cache_data()
def load_data():
    with open('./data/final_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    train_df_sample = pd.read_csv(GLOBAL_PATH + TRAIN_DATASET_SAMPLE)
    #train_df = pd.read_csv(GLOBAL_PATH + TRAIN_DATASET)

    return model, train_df_sample

model, train_df_sample = load_data()

features = [col for col in train_df_sample.columns if col not in ['target_Yes', 'transaction_id', 'transaction_date', 
                                                           'acct_open_date', 'year_pin_last_changed', 'card_expiration_date', 'mcc_description']]
target = 'target_Yes'

X_sample = train_df_sample[features].select_dtypes(include=[np.number])  # Focus on numeric for a start
y_sample = train_df_sample[target]

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)
y_pred = model.predict(X_test)

st.markdown('## Model Predcitions')
st.markdwon("""
The model has made predictions on the test data. Below are the first 5 predictions.
""")
st.write("Predictions:", y_pred[:5])


