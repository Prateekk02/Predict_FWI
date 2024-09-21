import pickle
import streamlit as st
import numpy as np
import pandas as pd
import os

# Set the directory where your model files are located
model_dir = os.path.join(os.getcwd(), 'models')

# Load ridge regressor and standard scaler pickle files using os module
ridge_model_path = os.path.join(model_dir, 'ridge.pkl')
scaler_model_path = os.path.join(model_dir, 'scaler.pkl')

# Make sure the files exist before loading
if os.path.exists(ridge_model_path) and os.path.exists(scaler_model_path):
    ridge_model = pickle.load(open(ridge_model_path, 'rb'))
    standard_scaler = pickle.load(open(scaler_model_path, 'rb'))
else:
    st.error("Model files not found. Please check the paths and ensure models exist.")
    
    
# Title and instructions
st.title('Fire Weather Index Prediction')
st.write('Enter the required values to predict the fire weather index.')

# Create input fields for the user to enter data
Temperature = st.number_input('Temperature', min_value=0.0, step=0.1, format="%.2f")
RH = st.number_input('Relative Humidity (RH)', min_value=0.0, step=0.1, format="%.2f")
Ws = st.number_input('Wind Speed (Ws)', min_value=0.0, step=0.1, format="%.2f")
Rain = st.number_input('Rainfall (Rain)', min_value=0.0, step=0.1, format="%.2f")
FFMC = st.number_input('FFMC Index', min_value=0.0, step=0.1, format="%.2f")
DMC = st.number_input('DMC Index', min_value=0.0, step=0.1, format="%.2f")
ISI = st.number_input('ISI Index', min_value=0.0, step=0.1, format="%.2f")
Classes = st.number_input('Classes', min_value=0.0, step=0.1, format="%.2f")
Region = st.number_input('Region', min_value=0.0, step=0.1, format="%.2f")

# Button to trigger the prediction
if st.button('Predict'):
    try:
        # Scale the input data
        new_scaled_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # Make the prediction using the ridge regression model
        result = ridge_model.predict(new_scaled_data)
        
        # Display the result
        st.success(f'The predicted Fire Weather Index is: {result[0]:.2f}')
    except Exception as e:
        st.error(f"Error: {e}")

