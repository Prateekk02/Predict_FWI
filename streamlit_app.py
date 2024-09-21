import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load ridge regressor and standard scaler pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

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

