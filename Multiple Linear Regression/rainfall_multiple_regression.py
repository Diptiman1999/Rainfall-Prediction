# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:56:14 2021

@author: dipti
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the dataset
dataset = pd.read_excel("RajgangpurNew.xlsx") 
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 8]

# Normalizing the data
from sklearn import preprocessing
X = preprocessing.normalize(X) 

# Splitting the dataset to training and test 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting the Mulltiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set result
y_pred = regressor.predict(X_test)

# To check for correctness of algorithm using r-score
from sklearn.metrics import r2_score
score = r2_score(Y_test,y_pred)
