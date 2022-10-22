# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import sklearn
import joblib,os
import numpy as np


#Loading Models

def load_prediciton_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model


def main():
    
    #Title
    st.title("Glucose Level Predictor")
    
    #Getting the input data
    
    VoltageLevel = st.text_input("Voltage Level")
    
    
    #Code for prediction
    
    
    if st.button("Predict"):
        
        regressor = load_prediciton_model("linear_regression_glucose_level.pkl")
        VoltageLevel_reshaped = np.array(VoltageLevel).reshape(-1,1)
        
        predicted_glucose_level = regressor.predict(VoltageLevel_reshaped)
        
        st.info("Glocuse level related to {} voltage : {}".format(VoltageLevel,predicted_glucose_level))
        
        

if __name__ == '__main__':
    main()
