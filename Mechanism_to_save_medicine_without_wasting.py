# Import Required Libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical visualizations
import joblib  # For loading the saved ML model
import streamlit as st  # For Streamlit app interface

# Load the trained XGBoost model
MODEL_PATH = "best_xgboost_model.pkl"  # Path to the saved model
model = joblib.load(MODEL_PATH)

# Define a function for predictions
def predict_demand(quantity, days_to_expiry, historical_demand):
    """
    Function to make predictions based on user inputs.
    """
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'quantity': [quantity],
        'days_to_expiry': [days_to_expiry],
        'historical_demand': [historical_demand]
    })
    
    # Make predictions using the loaded model
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit App Layout
st.title("Medicine Demand Prediction App 🚀")
st.write("""
### Save Medicines Without Wasting!  
This app predicts the medicine demand based on quantity, days left to expiry, and historical demand.
""")

# Sidebar for User Inputs
st.sidebar.header("Enter Medicine Details:")
quantity = st.sidebar.slider("Quantity of Medicine Boxes", 10, 200, 50, step=10)
days_to_expiry = st.sidebar.slider("Days Left to Expiry", 1, 365, 90, step=5)
historical_demand = st.sidebar.slider("Historical Demand", 10, 150, 50, step=5)

# Predict Button
if st.sidebar.button("Predict Demand"):
    prediction = predict_demand(quantity, days_to_expiry, historical_demand)
    
    # Display the Results
    st.success(f"📊 Predicted Medicine Demand: **{prediction:.2f} units**")

    # Visualize the Inputs and Prediction
    st.subheader("Inputs and Prediction Visualized 📈")
    fig, ax = plt.subplots()
    bars = ['Quantity', 'Days to Expiry', 'Historical Demand', 'Predicted Demand']
    values = [quantity, days_to_expiry, historical_demand, prediction]
    colors = ['blue', 'green', 'orange', 'purple']
    plt.bar(bars, values, color=colors)
    plt.title("Inputs and Predicted Demand")
    plt.ylabel("Values")
    st.pyplot(fig)
else:
    st.info("👈 Adjust the sliders and click 'Predict Demand' to get results!")

# Footer
st.write("#### Built with ❤️ using Streamlit and XGBoost")
