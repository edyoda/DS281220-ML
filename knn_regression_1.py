# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:17:27 2021

@author: RISHBANS
"""

import pandas as pd

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
nn_model = KNeighborsRegressor(n_neighbors = 3)
nn_model.fit(X_train, y_train)

y_pred = nn_model.predict(X_test)
print(y_pred)
print(nn_model.score(X_train, y_train))
print(nn_model.score(X_test, y_test))


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = []

for k in range(2, 20):
    nn_model = KNeighborsRegressor(n_neighbors = k)
    nn_model.fit(X_train, y_train)
    y_predict = nn_model.predict(X_test)
    
    error = sqrt(mean_squared_error(y_test, y_predict))
    rmse.append(error)
    print(k, error)
    
graph = pd.DataFrame(rmse)
graph.plot()    











