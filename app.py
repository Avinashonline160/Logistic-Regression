import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Purchase Prediction",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Life Insurance Purchase Prediction")
st.write("Predict whether a person will buy life insurance based on their age using Logistic Regression.")

# -----------------------------------
# Load Dataset
# -----------------------------------
# Make sure "insurance_data.csv" is in the same directory as this file
try:
    df = pd.read_csv("insurance_data.csv")
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("Error: 'insurance_data.csv' not found. Please place the dataset in the same directory.")
    st.stop()

# -----------------------------------
# Train Model
# -----------------------------------
X = df[['age']]
y = df['bought_insurance']

model = LogisticRegression()
model.fit(X, y)

# -----------------------------------
# User Input
# -----------------------------------
st.subheader("Enter Customer Details")

age = st.number_input(
