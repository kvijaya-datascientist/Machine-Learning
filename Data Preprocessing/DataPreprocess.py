# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 22:47:08 2020

@author: VijayaKaja
"""
#Data Preprocessing

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading the Data
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:,-1].values

#Handling missing values in dataset
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer= imputer.fit(x[: , 1:3])
x[:,1:3] = imputer.transform(x[:,1:3]) """
#    OR
from sklearn.impute import SimpleImputer
missingValues = SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
missingValues = missingValues.fit(x[:,1:3])
x[:,1:3] = missingValues.fit_transform(x[:,1:3])

#Encoding Categorical data(x,y)
"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder_x = OneHotEncoder(categorical_features=[0])
X=onehotencoder_x.fit_transform(x).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(y)  """

#   OR

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
Y = LabelEncoder().fit_transform(y)   # Encoding y dataset data
from sklearn.compose import ColumnTransformer
x[:,0] = LabelEncoder().fit_transform(x[:,0]) #Encoding x data
ct = ColumnTransformer([('encoder',OneHotEncoder(categorical_features=[0]), [0])],remainder='passthrough')
X = np.array(ct.fit_transform(x),dtype=np.float) # with Dummy Variables

#splitting the dataset into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().transform(x_test)








