# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:57:23 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Largecap_balancesheet.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("City" , OneHotEncoder(), [4])], remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(lr.score(X_test, y_test))

import matplotlib.pyplot as plt

x_serial = list(range(1, len(y_pred) + 1))
plt.scatter(x_serial, y_pred, color = 'red')
plt.scatter(x_serial, y_test, color = 'blue')
plt.title('y_pred vs y_actual')
plt.xlabel('serial number')
plt.ylabel('Profit')
plt.show()
