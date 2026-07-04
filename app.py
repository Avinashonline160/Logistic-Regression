import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

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
# Load Dataset & Train Model (Cached & Path-Safe)
# -----------------------------------
@st.cache_data
def load_and_train():
    # Dynamic path handling to fix Streamlit Cloud's FileNotFoundError
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "houseprice.csv")
    
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Extract features and target variable
    # Using double brackets [['area']] ensures X stays a 2D array/DataFrame
    X = df[["area"]] 
    y = df["price"]
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    return df, model

# Execute the data loading and training
try:
    df, model = load_and_train()
except FileNotFoundError:
    st.error("""
        **Crucial Error: `houseprice.csv` not found!** Please ensure that the file `houseprice.csv` is uploaded to your GitHub repository in the exact same folder as this `app.py` file.
    """)
    st.stop()

# -----------------------------------
# Display Dataset Preview
# -----------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

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
# Prediction Logic
# -----------------------------------
if st.button("Predict Price"):
    # Convert input into a DataFrame with identical feature name ('area')
    input_data = pd.DataFrame([[area]], columns=["area"])
    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Model Details")

st.write(f"**Coefficient:** {model.coef_[0]:.2f}")
st.write(f"**Intercept:** {model.intercept_:.2f}")
    
