import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

GLOBAL_PATH = './data/'
TRAIN_DATASET_SAMPLE = "train_df_sample.csv"
TRAIN_DATASET = "train_df.csv"

@st.cache_data()
def load_data():
    with open('./data/final_model.pkl', 'rb') as file:
        model_dict = pickle.load(file)  # Load the model dictionary
    
    # Extract the actual model from the dictionary (update 'model' key as needed)
    model = model_dict.get('model')
    
    if model is None:
        raise ValueError("No model found in the dictionary")
    
    train_df_sample = pd.read_csv(GLOBAL_PATH + TRAIN_DATASET_SAMPLE)
    return model, train_df_sample

# Load the model and data
model, train_df_sample = load_data()

# Split the data into features (X) and target (y)
X = train_df_sample.drop(labels=['target_Yes'], axis=1)
y = train_df_sample['target_Yes']

# Split the data into training and validation sets
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict using the loaded model
y_pred = model.predict(X_train_full)

# Streamlit UI
st.markdown('## Model Predictions')
st.markdown("""
The model has made predictions on the training data. Below are the first 5 predictions.
""")
st.write("Predictions:", y_pred[:5])
st.write("Corresponding Values:", X_train_full[:5])
