# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:04:55 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Company_Profit.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.score(X_train, y_train))

y_pred = lr.predict(X_train)

#Training Set
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_train, y_pred, color = 'red')
plt.scatter(X_train, y_pred, color = 'blue')
plt.title("training set")
plt.xlabel("years in operation")
plt.ylabel("profit")
plt.show()

#Test Set
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_test, lr.predict(X_test), color = 'red')
plt.scatter(X_test, lr.predict(X_test), color = 'blue')
plt.title("Test Set")
plt.xlabel("years in operation")
plt.ylabel("profit")
plt.show()
















