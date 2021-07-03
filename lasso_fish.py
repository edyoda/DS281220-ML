# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:18:14 2021

@author: RISHBANS
"""

#1. Import the dataset
import pandas as pd
dataset = pd.read_csv('Fish.csv')
#2. Store into X and y
X = dataset.iloc[:, 2:7].values
y = dataset.iloc[:, 1].values

#3. Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state = 0)

#4. Apply Standard Scaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#5. Model = Linear Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print(lr_model.score(X_train, y_train))
print(lr_model.score(X_test, y_test))

#6. Lasso
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 2)
lasso_model.fit(X_train, y_train)
print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test))

#7. Ridge
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha = 2)
ridge_model.fit(X_train, y_train)
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))










