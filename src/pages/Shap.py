import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
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


X = train_df_sample.drop(labels=['target_Yes'], axis=1)
y = train_df_sample['target_Yes']

# Split the data into training and validation sets
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Title of the SHAP analysis page
st.title("SHAP Values - Model Explainability")

# Select sample data for SHAP
sample_for_shap = X_val_full.sample(1000, random_state=42)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_for_shap)

# Summary plot (Bar)
st.markdown("### SHAP Summary Plot (Bar)")
st.markdown("""
We observe in the plot what are the different impacts in the model output of various features. 
""")
shap.summary_plot(shap_values, sample_for_shap, plot_type="bar")
st.pyplot(plt.gcf())

# SHAP Summary plot (dot)
st.markdown("### SHAP Summary Plot (Dot)")
st.markdown("""
This visualization mixes the different ranges of values that the features can take and their impact on the model output.
We can observe 3 interesting features, transaction_hour, mcc and transaction_amount, we also could consider the transaction_year but is not relevant in this case because of how the df is constructed where only 2 years are considered [2010] (low information). 
We will observe them individually for further analysis. 
""")
shap.summary_plot(shap_values, sample_for_shap)
st.pyplot(plt.gcf())

st.markdown("""
To understand how a single feature’s value influences predictions in conjunction with another feature we will use a dependence plot using the principal features that had more impact on the model output:
""")
# Selected features for dependence plots
features = ["transaction_hour", "mcc", "transaction_amount"]
for i in features:
    st.markdown(f"### SHAP Dependence Plot for {i}")
    shap.dependence_plot(i, shap_values, sample_for_shap)
    st.pyplot(plt.gcf())

st.markdown("""
Transaction_hour vs. SHAP value:
    We observe that mid-range hours (e.g., around midday) might show higher or lower SHAP values, suggesting that certain hours increase the likelihood of fraud. The color encoding reveals that this hour-based pattern evolves over time. 

Mcc vs. SHAP value:
    The mcc plot shows how certain categories are associated with positive or negative contributions to fraud likelihood. The second color dimension shows that within certain MCC ranges, higher transaction amounts push predictions in a particular direction.

Transaction_amount vs. SHAP value:
    By plotting SHAP values against transaction_amount and coloring by transaction_hour, we might see that large transaction amounts during particular hours have a stronger influence on the fraud prediction. Lower amounts may cluster around zero SHAP impact, while extreme values (either very high or unusual amounts) create strong positive or negative pushes in the model’s output.
""")

# Compute interaction values
shap_interaction_values = explainer.shap_interaction_values(sample_for_shap)

# SHAP Interaction summary plot
st.markdown("### SHAP Interaction Summary Plot")
shap.summary_plot(shap_interaction_values, sample_for_shap)
st.pyplot(plt.gcf())

st.markdown("""
On this interaction plot we observe how pairs of features jointly affect the prediction.
""")