# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import sklearn
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

#Loading Models

loaded_model = pickle.load(open('trained_model.sav','rb'))
scaler = pickle.load(open('standardized_data.pkl','rb'))

def glucose_prediction(input_data):
    
    input_data_as_npArray = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_npArray.reshape(-1,1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    return prediction[0]


def main():
    
    #Title
    st.title("Glucose Level Predictor")
    
    #Getting the input data
    
    VoltageLevel = st.text_input("Voltage Level")
    
    
    #Code for prediction
    
    diagnosis = ''
    
    if st.button("Predict"):
        
        diagnosis = glucose_prediction(VoltageLevel)
        
    st.success(diagnosis)
        
        

if __name__ == '__main__':
    main()
