import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

GLOBAL_PATH = './data/'
TRANSACTIONS_WITH_LABELS_DATASET = "transactions_with_labels.csv"
MCC_CODES_CLEAN_DATASET = "mcc_codes_df_clean.csv"
USERS_CLEAN_DATASET = "users_df_clean.csv"
CARDS_CLEAN_DATASET = "cards_df_clean.csv"

@st.cache_data()
def load_data():
    transactions_with_labels = pd.read_csv(GLOBAL_PATH + TRANSACTIONS_WITH_LABELS_DATASET)
    mcc_codes_df_clean = pd.read_csv(GLOBAL_PATH + MCC_CODES_CLEAN_DATASET)
    users_df_clean = pd.read_csv(GLOBAL_PATH + USERS_CLEAN_DATASET)
    cards_df_clean = pd.read_csv(GLOBAL_PATH + CARDS_CLEAN_DATASET)
    return transactions_with_labels, mcc_codes_df_clean, users_df_clean, cards_df_clean

transactions_with_labels, mcc_codes_df_clean, users_df_clean, cards_df_clean = load_data()

cards_df_clean['card_expiration_date'] = pd.to_datetime(cards_df_clean['card_expiration_date'])
cards_df_clean['acct_open_date'] = pd.to_datetime(cards_df_clean['acct_open_date'])
cards_df_clean['year_pin_last_changed'] = pd.to_datetime(cards_df_clean['year_pin_last_changed'])

st.subheader("Exploratory Data Analysis and Insights")

transaction_counts = transactions_with_labels['target_Yes'].value_counts()
fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1.pie(transaction_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'red'])
ax1.set_title('Distribution of Fraud vs Non-Fraud Transactions')
st.markdown("""
The dataset turns out to be very unbalanced as 0.2% \of transactions are fraudulent and the other 98.8% \are non fraudulent.
""")
st.pyplot(fig1)

# Transaction Amounts for Fraud vs Non-Fraud Transactions
st.subheader("Transaction Amounts for Fraud vs Non-Fraud Transactions")
st.markdown("""
The plot shows a lot of outliers are found in the dataset, However, after removing those by computing the IQR, we see that the fraudulent transactions have a slightly higher mean (38$ Fraud vs 37$ Non-Fraud).
""")
fig2, ax2 = plt.subplots()
sns.violinplot(x='target_Yes', y='transaction_amount', data=transactions_with_labels, ax=ax2)
ax2.set_title('Transaction Amounts for Fraud vs Non-Fraud Transactions')
st.pyplot(fig2)

# Filtering Outliers for Transaction Amount
st.subheader("Filtered Transaction Amount Analysis")
st.markdown("""
This plot shows that the most likely places where fraud takes place are grocery stores, supermarkets, food stores and service stations. 
""")

Q1 = transactions_with_labels['transaction_amount'].quantile(0.25)
Q3 = transactions_with_labels['transaction_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_transactions = transactions_with_labels[
    (transactions_with_labels['transaction_amount'] >= lower_bound) & 
    (transactions_with_labels['transaction_amount'] <= upper_bound)
]
mean_transaction_amount_by_target = filtered_transactions.groupby('target_Yes')['transaction_amount'].mean()
st.write("Mean Transaction Amount by Target (Filtered):", mean_transaction_amount_by_target)

# Fraud by MCC Description
st.subheader("Fraud Transactions by MCC Description")
st.markdown("""
This plot shows that the most likely places where fraud takes place are grocery stores, supermarkets, food stores and service stations. 
""")
transactions_with_labels_mcc = transactions_with_labels.merge(mcc_codes_df_clean, on='mcc', how='left')
fraud_by_region = transactions_with_labels_mcc[transactions_with_labels_mcc['target_Yes'] == 1] \
    .groupby('mcc_description')['target_Yes'].count() \
    .sort_values(ascending=False)

min_fraud_threshold = 100
fraud_by_region_filtered = fraud_by_region[fraud_by_region >= min_fraud_threshold]

plt.figure(figsize=(12, 6))
fraud_by_region_filtered.plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.title('Fraud Transactions by MCC Description')
plt.tight_layout() 
st.pyplot(plt)

# Fraudulent Transactions by Hour
st.subheader("Fraudulent Transactions by Hour of the Day")
st.markdown("""
This graph shows that most of the Fraud Transactions happen between 11 and 12 AM.  
""")
transactions_with_labels['transaction_date'] = pd.to_datetime(transactions_with_labels['transaction_date'], format="%Y-%m-%d %H:%M:%S")
fraud_transactions = transactions_with_labels[transactions_with_labels['target_Yes'] == 1].copy()
fraud_transactions['hour'] = fraud_transactions['transaction_date'].dt.hour
fig4, ax4 = plt.subplots()
sns.countplot(x='hour', data=fraud_transactions, ax=ax4)
ax4.set_title('Fraudulent Transactions by Hour of the Day')
st.pyplot(fig4)

# Correlation Heatmap
st.subheader("Correlation Heatmap of Numeric Features")
st.markdown("""
The correlation matrix showed that any of the variables had any positive nor negative correlation with the target variable. 
""")
correlation_df = transactions_with_labels.merge(users_df_clean, on='user_id', how='left')
correlation_df = correlation_df.merge(cards_df_clean, on='card_id', how='left')
correlation_df = correlation_df.drop(columns=['transaction_id', 'user_id_x', 'user_id_y', 'card_id', 'birth_year', 'birth_month', 'latitude', 'longitude'])
corr = correlation_df.corr()
fig5, ax5 = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_df.corr()[['target_Yes']], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax5)
ax5.set_title('Correlation Heatmap of Numeric Features')
st.pyplot(fig5)
