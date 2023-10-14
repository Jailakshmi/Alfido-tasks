# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:14:23 2023

@author: Jai
"""
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

pickle_in = open("knn1.pkl","rb")
knn1 = pickle.load(pickle_in)

prediction = knn1.predict([[4.9,3.0,1.4,0.2]]) # sep_len, sep_wid, petal_len, petal_wid

print(prediction[0])   # Iris-setosa : 0, Iris-versicolor : 1, Iris-virginica : 2
if prediction[0]:
    print('Iris-versicolor')
elif prediction[0]==0:
    print('Iris-setosa')
else:
    print('Iris-virginica')
