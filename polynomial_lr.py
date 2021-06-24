# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:24:37 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Company_Performance.csv")

X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values


from sklearn.linear_model import LinearRegression
simple_lr = LinearRegression()
simple_lr.fit(X, y)
print(simple_lr.score(X, y))


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

poly_lin_reg = LinearRegression()
poly_lin_reg.fit(X_poly, y)
y_pred = poly_lin_reg.predict(X_poly)
print(poly_lin_reg.score(X_poly, y))

import matplotlib.pyplot as plt
#Simple Linear Regression
plt.scatter(X, y , color = 'red')
plt.plot(X, simple_lr.predict(X), color = 'blue')
plt.title('Size of Company(Simple Linear Regression')
plt.xlabel('No. of years')
plt.ylabel('No of Emp')
plt.show()

#Polynomial Linear Regression
plt.scatter(X, y , color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Size of Company(Polynomial Linear Regression- 4')
plt.xlabel('No. of years')
plt.ylabel('No of Emp')
plt.show()













