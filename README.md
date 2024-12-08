# Project AV

## Fraud Detection System Using Data Visualization and Machine Learning

### Norbert Tomàs Escudero (242695) Marc Sirvent Ruiz (240198)


## Instructions

The project contains a python notebook ```main.ipynb``` which contains all the data cleaning, EDA, model training and SHAP. To run the data cleaning part, a folder named ```data``` has to be created with the original datasets which can be found in ```https://drive.google.com/drive/folders/1zgpJxfizyLs36syczhZJ61LKPopddv4q?usp=sharing``` or in the original Kaggle post ```https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets?select=train_fraud_labels.json```. 

In the folder drive folder ```https://drive.google.com/drive/folders/1zgpJxfizyLs36syczhZJ61LKPopddv4q?usp=sharing``` the necessary datasets to run the Streamlit application can be found. 

The streamlit application can be run by typing ```streamlit run "./src/Home Page.py"```. Make sure you are in the correct directory and you have the necessary files downloaded before running.

For any question, please contact:
- norbert.tomas01@estudiant.upf.edu
- marc.sirvent01@estudiant.upf.edu


## Project overview

The goal of this project is to build an integrated solution for detecting and preventing fraudulent financial transactions.

Our project focuses on answering the following questions:
- What trends and patterns are evident in transaction data, and how do they relate to fraudulent activity? (difference in trends and patterns from transaction data (non-fraudulent) and fraudulent)  
- Which features (e.g., transaction amount, card type, country) are the most important indicators of fraud? (Using SHAP on our predictive model of fraud) 
- How can we present insights and predictions in a user-friendly manner for decision-making? 

The results of this project will assist stakeholders in identifying fraud patterns through data-driven decisions.

Dataset extracted from ```https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets?select=train_fraud_labels.json```.