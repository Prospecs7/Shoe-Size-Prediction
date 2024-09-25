import streamlit as st
import numpy as np
import joblib


st.title('Shoe Size Prediction Model')

#Load Models
#scaler = joblib.load('Pscaler.pkl')
forest = joblib.load('Pforest.pkl')
    

height = st.number_input('Enter the height in cm')
gender = st.radio("Select Gender", ["Female", "Male"])
gender_encoded = 0 if gender == 'Female' else 1

features = np.array([[height, gender_encoded]])

if st.button('Predict'):
    #scaled_f = scaler.fit_transform(features)
    

    prediction = forest.predict(features)
    
    
    predicted_size = round(prediction[0])


    st.write(f'Predicted Shoe Size: {predicted_size}')
    
