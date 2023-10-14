# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:14:56 2023

@author: Jai
"""


import streamlit as st
import pickle
import numpy as np

loaded_model2 = pickle.load(open(r'E:\docs jai\Intern_tasks\iris\knn1.pkl','rb'))

def prediction(input_data):
    
     input_array = np.asarray(input_data)
     input_reshaped = input_array.reshape(1,-1)
     prediction = loaded_model2.predict(input_reshaped)
     print(prediction[0])   # Iris-setosa : 0, Iris-versicolor : 1, Iris-virginica : 2
     if prediction[0]:
          return 'Iris-versicolor'
     elif prediction[0]==0:
          return 'Iris-setosa'
     else:
          return 'Iris-virginica'
 
def main():
    st.title('Iris flower classifying model')    
    
    Sepal_length = st.number_input('Sepal length')
    Sepal_width = st.number_input('Sepal width')
    Petal_length = st.number_input('Petal length')
    Petal_width = st.number_input('Petal width')
    
    pred = ''
    if st.button('Find the species of the flower'):
        pred = prediction([Sepal_length,Sepal_width,Petal_length,Petal_width])
        
    st.success(pred)
    
    
if __name__ == '__main__':
        main()    