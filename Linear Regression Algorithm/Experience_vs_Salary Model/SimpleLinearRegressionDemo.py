# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:19:48 2020

@author: VijayaKaja
"""

#Simple Linear Regression Algorithm

#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading the data
dataset = pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Splitting the data into train & test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#fitting simple linear regression model to train dataset
from sklearn.linear_model import LinearRegression
reg_train = LinearRegression().fit(x_train,y_train)

#Predecting y_test results using model
pred_ytest = reg_train.predict(x_test)

#Predecting  the salary for 9 & 12 years experience
y9 = reg_train.predict([[9]])
y12 = reg_train.predict([[12]])

#Visualizing the train set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg_train.predict(x_train),color='blue')
plt.title('Experience vs Salary(train set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg_train.predict(x_train),color='blue')
plt.title('Experience vs Salary(test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

