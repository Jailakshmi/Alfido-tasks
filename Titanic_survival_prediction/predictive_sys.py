# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:30:38 2023

@author: Jai
"""

import pickle
import numpy as np

loaded_model = pickle.load(open(r'E:\docs jai\Intern_tasks\titanic\trained_model.sav','rb'))


input_data = (20,1,3,0) # age, sex, Pclass, Embarked
input_array = np.asarray(input_data)
input_reshaped = input_array.reshape(1,-1)
prediction = loaded_model.predict(input_reshaped)
print(prediction[0])
if prediction[0]:
    print('The passenger has survived')
else:
    print('The passenger has not survived')