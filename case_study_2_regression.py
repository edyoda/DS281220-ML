# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 19:50:20 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Fish.csv")

corr = dataset.corr()
del dataset['Length2']
del dataset['Length3']

#Split into X and y
X = dataset.iloc[:, 2:5].values
y = dataset.iloc[:, 1].values


#Split into Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state = 0)

#Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))
