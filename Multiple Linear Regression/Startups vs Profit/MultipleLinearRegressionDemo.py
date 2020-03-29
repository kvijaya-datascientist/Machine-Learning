# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:06:51 2020

@author: VijayaKaja
"""
#Multiple Linear Regression

#Taken startup's information & have to predict the profit by state

#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding the independent variable(State)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
x[:,3] = LabelEncoder().fit_transform(x[:,3])
x= OneHotEncoder(categorical_features=[3]).fit_transform(x).toarray()

#Splitting the dataset into train & test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#fitting Multiple Linear Regression to train set
from sklearn.linear_model import LinearRegression
reg_train = LinearRegression().fit(x_train,y_train)

ypred = reg_train.predict(x_test)


