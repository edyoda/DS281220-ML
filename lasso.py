# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:19:59 2021

@author: RISHBANS
"""

import pandas as pd
import numpy as np
auto_data = pd.read_csv("Detail_Cars.csv")

auto_data = auto_data.replace('?', np.nan)
col_obj = auto_data.select_dtypes(include=["object"])
print(col_obj)
auto_data['price'] = pd.to_numeric(auto_data['price'], errors = 'coerce')
auto_data['bore'] = pd.to_numeric(auto_data['bore'], errors = 'coerce')
auto_data['stroke'] = pd.to_numeric(auto_data['stroke'], errors = 'coerce')
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors = 'coerce')
auto_data['peak-rpm'] = pd.to_numeric(auto_data['peak-rpm'], errors = 'coerce')

auto_data = auto_data.drop('normalized-losses', axis = 1)
cylin_dict = {'two': 2, 'three':3, 'four':4, 'five':5, 'six':6,
              'eight':8, 'twelve':12}
auto_data['num-of-cylinders'].replace(cylin_dict, inplace=True)

auto_data = pd.get_dummies(auto_data, drop_first = True)

auto_data = auto_data.dropna()

corr = auto_data.corr()

X = auto_data.drop('price', axis = 1)
y = auto_data['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

predictors = X_train.columns
coef = pd.Series(model.coef_, predictors).sort_values()

coef.plot(kind='bar')

from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 2, normalize = True)
lasso_model.fit(X_train, y_train)

print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test))

predictors = X_train.columns
lasso_coef = pd.Series(lasso_model.coef_, predictors).sort_values()

lasso_coef.plot(kind='bar')

y_pred = lasso_model.predict(X_test)

import matplotlib.pyplot as plt
plt.plot(y_pred, label='predict')
plt.plot(y_test.values, label = 'actual')
plt.ylabel('lasso')
plt.legend()
plt.show()


from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=2, normalize = True)
ridge_model.fit(X_train, y_train)
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))

predictors = X_train.columns
ridge_coef = pd.Series(ridge_model.coef_, predictors).sort_values()

ridge_coef.plot(kind='bar')


































