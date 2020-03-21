# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:10:24 2020

@author: VijayaKaja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_dataset = pd.read_csv('Position_Salaries.csv')
x=df_dataset.iloc[:,1:2:3].values
y=df_dataset.iloc[:,2].values

#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

#predict the salaries based on level
y8level_sal=regressor.predict([[8]])
y14level_sal = regressor.predict([[14]])
y3level_sal = regressor.predict([[3]])

#Visualizing the Random Forest Regression Results
plt.scatter(x,y,color='blue')
plt.plot(x,regressor.predict(x),color='red')
plt.title('Position vs Salary (RFR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Random Forest Regression Results
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='blue')
plt.plot(x_grid,regressor.predict(x_grid),color='green')
plt.title('Position Level vs Salary(RFR)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


