import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#streamlit run "./src/Home Page.py"

# Title and Introduction
st.title("Project AV: Fraud Detection System Using Data Visualization and Machine Learning")

# Add project introduction with team members
st.markdown("""
### Team Members
- **Norbert Tomàs Escudero (242695)**  
- **Marc Sirvent Ruiz (240198)**

---

### Project Overview
The goal of this project is to build an integrated solution for detecting and preventing fraudulent financial transactions. 

Our project focuses on answering the following key questions:
1. **What trends and patterns are evident in transaction data, and how do they relate to fraudulent activity?**  
   - We analyze differences in trends and patterns between non-fraudulent and fraudulent transactions.
   - Visualizations like heatmaps, bar charts, and time series help uncover insights.

2. **Which features are the most important indicators of fraud?**  
   - Using advanced techniques like SHAP (SHapley Additive exPlanations), we identify key features such as transaction amount, card type, and country.

3. **How can we present insights and predictions in a user-friendly manner for decision-making?**  
   - Through intuitive visualizations and interactive dashboards to facilitate insights.

---

### Objective
This project aims to assist stakeholders in identifying fraud patterns through data-driven decisions, providing actionable insights for fraud prevention and detection.

Explore the tabs above for detailed analysis, visualizations, and our predictive fraud model.

---

### Problem Statement
As the data analytics team of an important banking institution, we have been tasked with building an integrated solution for detecting and preventing fraudulent financial transactions. With financial institutions facing growing threats from fraud, it's essential to create reliable, automated methods to identify fraudulent activity before it results in significant losses.

The outcome of this project will help stakeholders identify fraud patterns and take data-driven actions to mitigate fraud risks. This could lead to more efficient fraud detection systems, better resource allocation, and ultimately, a safer banking environment for customers.

---

### Dataset Overview:
The banking institution has provided us with specific data containing the following information:
- **Transaction Data**: Transaction records including amounts, timestamps, and merchant details.
- **Card Information**: Credit and debit card details including card limits, types, and activation dates.
- **User Data**: Demographic information about customers and account-related details.
- **Fraud Labels**: Binary classification labels for transactions, indicating fraudulent vs. legitimate transactions.
- **Merchant Category Codes (MCC)**: Standard classification codes for business types, useful to identify each code in the transaction data mcc column.

---

### Business Questions and Objectives:
Our project addresses the following key questions:

1. **What trends and patterns are evident in transaction data, and how do they relate to fraudulent activity?**
   - Identify patterns linked to fraud, such as outliers, anomalies, and trends not common in legitimate transactions.

2. **Which features (e.g., transaction amount, card type, country) are the most important indicators of fraud?**
   - Identifying and quantifying features like transaction amount, merchant type, and transaction origin that help predict fraud.

3. **How can we present insights and predictions in a user-friendly manner for decision-making?**
   - Present findings in an easily digestible format for stakeholders like bank managers and fraud analysts to make informed decisions.

---

Move between the different pages on the left to visualize the different insights and model results of this project. 
         
---
""")

# Display a Data Preview Section
st.markdown("### Data Preview")
st.markdown("""
Here is a sample of the data used for fraud detection. This dataset includes both fraudulent and legitimate transactions with various features such as transaction amount, merchant details, and card information.

""")
# Assuming the data has already been loaded into train_df_sample
train_df_sample = pd.read_csv('./data/train_df_sample.csv')  # Example file, update with your file path

st.write(train_df_sample.head())  # Show a preview of the dataset
