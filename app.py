import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict house price using Linear Regression")

# -----------------------------------
# Load Dataset & Train Model (Cached)
# -----------------------------------
# We use st.cache_data so this ONLY runs once, making your app lightning fast!
@st.cache_data
def load_and_train():
    df = pd.read_csv("houseprice.csv")
    
    # Force X to be ONLY the 'area' column to match your single input slider
    # If your column is named differently, change "area" to match your CSV
    X = df[["area"]] 
    y = df["price"]
    
    model = LinearRegression()
    model.fit(X, y)
    return df, model

# Load data and model
df, model = load_and_train()

# -----------------------------------
# Display Dataset
# -----------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head()) # .head() displays the first 5 rows so the page stays clean

# -----------------------------------
# User Input
# -----------------------------------
st.subheader("Enter House Area")

area = st.number_input(
    "Area (Square Feet)",
    min_value=100,
    max_value=10000,
    value=3300,
    step=100
)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Price"):
    # Features must be passed as a 2D array/DataFrame with matching feature names
    input_data = pd.DataFrame([[area]], columns=["area"])
    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Model Details")

st.write(f"**Coefficient:** {model.coef_[0]:.2f}")
st.write(f"**Intercept:** {model.intercept_:.2f}")
