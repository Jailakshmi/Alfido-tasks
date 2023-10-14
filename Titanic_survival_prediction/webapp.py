# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:50:34 2023

@author: Jai
"""

import streamlit as st
import pickle
import numpy as np

loaded_model = pickle.load(open(r'E:\docs jai\Intern_tasks\titanic\trained_model.sav','rb'))

def prediction(input_data):
    
    input_array = np.asarray(input_data)
    input_reshaped = input_array.reshape(1,-1)
    prediction = loaded_model.predict(input_reshaped)
    print(prediction[0])
    if prediction[0]:
        return 'The passenger has survived'
    else:
        return 'The passenger has not survived'


def main():
    st.title('Titanic survival prediction model')  

    
    Age = st.number_input('Age of the passenger')
    Sex = st.number_input('Sex of the passenger (M:1, F:0)')
    Pclass = st.number_input('Passenger class of the passenger (1, 2 or 3)')
    Embarked = st.number_input('Embarking of the passenger(C:0, Q:1 or S:2 )')
    input_data = ([Age,Sex,Pclass,Embarked]) # age, sex, Pclass, Embarked
    
    pred = ''
    if st.button('Predict survival'):
        pred = prediction(input_data)
        
    st.success(pred)
    
    
if __name__ == '__main__':
        main()    
        
        

              