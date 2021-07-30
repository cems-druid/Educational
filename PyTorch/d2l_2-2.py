# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:08:27 2021

@author: cemsi
"""

#2-2 Data preprocessing 

#Creating the file and writing into
import os

os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n')
    f.write('NA, Pave, 127500\n')
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')
    
#Reading the csv file with pandas library
import pandas as pd

data = pd.read_csv(data_file)
#print(data)

#Handling na data, by imputation (changing with mean of similar ones) of numerical data, for categorical
#one-hot encoding is done
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
#print(inputs)

inputs = pd.get_dummies(inputs, dummy_na = False)
#print(inputs)

#Conversion into tensor format

import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)

