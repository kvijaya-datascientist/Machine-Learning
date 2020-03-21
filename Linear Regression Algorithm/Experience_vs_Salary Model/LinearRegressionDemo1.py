# Usecase - Salary expetation based on years of experience : Simple linear Regression 

# -*- coding: utf-8 -*-
"""
@author : kviajya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_dataset = pd.read_csv('Salary_Data.csv')
# divivde the dataset into x & y
x = df_dataset.iloc[:,:-1].values # OR [:,:-1]  
y = df_dataset.iloc[:,1].values
# divide the dataset into train & test datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
#implement our classifier based on simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results
y_predict=regressor.predict(x_test)

#predicting the salary with 11 years & 9 years experience

y9_sal = regressor.predict([[9]])
y11_sal= regressor.predict([[11]])

#visualizing the training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')

#visualizing the test set results
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,regressor.predict(y_test),color='green')
plt.title('Simple Line Regression-Salary vs Experience Test')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()




