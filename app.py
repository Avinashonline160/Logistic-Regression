import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Purchase Prediction",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insurance Purchase Prediction")
st.write("Predict if a customer will buy insurance based on their age using Logistic Regression.")

# -----------------------------------
# Load Dataset & Train Model (Cached)
# -----------------------------------
@st.cache_data
def load_and_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "insurance_data.csv") # Make sure this matches your CSV filename on GitHub!
    
    df = pd.read_csv(csv_path)
    
    # Split features and target
    X = df[['age']]
    y = df.bought_insurance
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return df, model

# Safe execution of data loading
try:
    df, model = load_and_train()
except FileNotFoundError:
    st.error("**Error:** The dataset CSV file could not be found. Please check your filename in GitHub.")
    st.stop()

# -----------------------------------
# Display Dataset Preview
# -----------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# User Input (Line 43 - Fixed!)
# -----------------------------------
st.subheader("Enter Customer Details")

age = st.number_input(
    "Age of the Customer",
    min_value=1,
    max_value=100,
    value=25,
    step=1
)

# -----------------------------------
# Prediction Logic
# -----------------------------------
if st.button("Predict"):
    # Create input DataFrame to match training format
    input_data = pd.DataFrame([[age]], columns=['age'])
    prediction = model.predict(input_data)
    
    # Predict probability for cleaner insight
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ Likely to buy insurance! (Probability: {probability:.2f}%)")
    else:
        st.error(f"❌ Unlikely to buy insurance. (Probability of buying: {probability:.2f}%)")
