#!/usr/bin/env python
# coding: utf-8

# ## Section 1: Import Libraries and Load Data

# In[2]:


# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualizations
import seaborn as sns  # For statistical plots
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.ensemble import RandomForestRegressor  # Random Forest model
from xgboost import XGBRegressor  # XGBoost model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Metrics for model evaluation
import joblib  # For saving and loading the model

# Load the dataset
file_path = "synthetic_medicine_inventory_data.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Display the first 5 rows of the dataset to understand its structure
print("First 5 rows of the dataset:")
print(data.head())

# Summary statistics of the dataset
print("\nDataset Overview:")
print(data.describe())


# ## Section 2: Exploratory Data Analysis (EDA)

# In[10]:


# Check for missing values in the dataset
print("\nMissing Values in the Dataset:")
print(data.isnull().sum())

# Plot distribution of 'quantity'
plt.figure(figsize=(10, 6))
sns.histplot(data['quantity'], kde=True, bins=20, color='blue')
plt.title('Distribution of Medicine Quantities')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

# Plot distribution of 'days_to_expiry'
plt.figure(figsize=(10, 6))
sns.histplot(data['days_to_expiry'], kde=True, bins=20, color='green')
plt.title('Distribution of Days to Expiry')
plt.xlabel('Days to Expiry')
plt.ylabel('Frequency')
plt.show()

# Distribution of Predicted Demand
plt.figure(figsize=(10, 6))
sns.histplot(data['predicted_demand'], kde=True, bins=20, color='purple')
plt.title('Distribution of Predicted Demand')
plt.xlabel('Predicted Demand')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Scatter Plot: Quantity vs Predicted Demand, colored by Days to Expiry
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='quantity',
    y='predicted_demand',
    hue='days_to_expiry',
    palette='cool',
    data=data
)


# Scatter plot between quantity and predicted demand, colored by days to expiry
plt.figure(figsize=(10, 6))
plt.scatter(data['quantity'], data['predicted_demand'], c=data['days_to_expiry'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Days to Expiry')
plt.title('Quantity vs Predicted Demand Colored by Days to Expiry')
plt.xlabel('Quantity')
plt.ylabel('Predicted Demand')
plt.grid(True)
plt.show()


# ## Section 3: Prepare Data for Training

# In[4]:


# Define features (X) and target variable (y)
features = ['quantity', 'days_to_expiry', 'historical_demand']  # Input columns
target = 'predicted_demand'  # Target column

X = data[features]  # Feature set
y = data[target]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the size of training and testing datasets
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


# ## Section 4: Train Linear Regression Model

# In[5]:


# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model using the training data
linear_model.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = linear_model.predict(X_test)

# Evaluate model performance
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Display Linear Regression performance
print("\nLinear Regression Model Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse_lr:.2f}")
print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"R-squared (R²): {r2_lr:.2f}")


# ## Section 5: Train Random Forest and XGBoost Models

# In[6]:


# Initialize Random Forest and XGBoost models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# Train Random Forest model
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Train XGBoost model
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate both models
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Display model performances
print("\nRandom Forest Model Performance:")
print(f"RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")

print("\nXGBoost Model Performance:")
print(f"RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.2f}")


# ## Section 6: Model Comparison Visualization

# In[7]:


# Create a summary table of model results
results = {
    "Linear Regression": [rmse_lr, r2_lr],
    "Random Forest": [rmse_rf, r2_rf],
    "XGBoost": [rmse_xgb, r2_xgb]
}

results_df = pd.DataFrame(results, index=["RMSE", "R²"]).T

# Display results
print("\nModel Performance Comparison:")
print(results_df)

# Visualize RMSE for each model
plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df['RMSE'], palette='viridis')
plt.title("Model Comparison: RMSE")
plt.ylabel("Root Mean Squared Error")
plt.show()


# ## Section 7: Save the Best Model

# In[8]:


# Save the XGBoost model as the best model (if it's performing best)
joblib.dump(xgb_model, "best_xgboost_model.pkl")
print("XGBoost model saved successfully as 'best_xgboost_model.pkl'")


# ## Section 8: Load Model and Make Predictions

# In[9]:


# Load the saved model
loaded_model = joblib.load("best_xgboost_model.pkl")
print("Model loaded successfully!")

# Example of new data for predictions
new_data = pd.DataFrame({
    'quantity': [120, 75, 90],
    'days_to_expiry': [45, 15, 100],
    'historical_demand': [100, 60, 80]
})

# Predict demand for new data
predictions = loaded_model.predict(new_data)
new_data['predicted_demand'] = predictions

# Display predictions
print("\nPredictions for New Data:")
print(new_data)


# In[11]:


from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
import joblib
loaded_model = joblib.load("best_xgboost_model.pkl")

# Define a function to predict and display results interactively
def predict_demand(quantity, days_to_expiry, historical_demand):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'quantity': [quantity],
        'days_to_expiry': [days_to_expiry],
        'historical_demand': [historical_demand]
    })
    
    # Predict using the loaded model
    predicted_demand = loaded_model.predict(input_data)[0]
    
    # Create a bar chart showing inputs and prediction
    plt.figure(figsize=(8, 5))
    plt.bar(['Quantity', 'Days to Expiry', 'Historical Demand', 'Predicted Demand'], 
            [quantity, days_to_expiry, historical_demand, predicted_demand],
            color=['blue', 'green', 'orange', 'purple'])
    
    # Add title and labels
    plt.title("Interactive Medicine Demand Prediction")
    plt.ylabel("Values")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate the predicted demand
    plt.text(3, predicted_demand + 2, f"{predicted_demand:.2f}", ha='center', fontsize=12, color='black')
    plt.show()
    
    # Print the predicted value as a summary
    print(f"📊 Predicted Demand: {predicted_demand:.2f} units")

# Create interactive sliders for input parameters
interact(
    predict_demand,
    quantity=IntSlider(min=10, max=200, step=10, value=50, description='Quantity 📦'),
    days_to_expiry=IntSlider(min=1, max=365, step=5, value=90, description='Days to Expiry ⏳'),
    historical_demand=IntSlider(min=10, max=150, step=5, value=50, description='Historical Demand 📊')
)


# In[ ]:




