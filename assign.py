# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 20:18:26 2021

@author: RISHBANS
"""

import pandas as pd
data = pd.read_csv("train.csv")
print(data.head())

def process_data(data):
    b = list(data.columns)
    data= data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'],axis = 1)
    
    b = list(data.columns)
    
    from sklearn import preprocessing
    #creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    for i in b:
        if data[i].dtype == "object":
            data[i]=le.fit_transform(data[i])
    return data
      
data = process_data(data)  
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

import numpy as np
from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values=np.nan, strategy='median')

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


data_2 = pd.read_csv("test.csv")
     
data_2 = process_data(data_2)
X_new = data_2.iloc[:,1:]

X_new1 = fill_0.transform(X_new)

var = model.predict(X_new1)
print(var)

