# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:40:06 2021

@author: RISHBANS
"""

from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()

#store values in dataframe
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['price'] = boston.target

#Split into X and y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#Perform standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Apply KNN
from sklearn.neighbors import KNeighborsRegressor
nn_model = KNeighborsRegressor(n_neighbors=2)
nn_model.fit(X_train, y_train)
y_pred = nn_model.predict(X_test)


#Chekc the train and test score
print(nn_model.score(X_train, y_train))
print(nn_model.score(X_test, y_test))

#Chekc the ideal value of k
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







