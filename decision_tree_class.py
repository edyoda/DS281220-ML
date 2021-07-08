# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:45:56 2021

@author: RISHBANS
"""

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
train_pred = dt.predict(X_train)
test_pred = dt.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, train_pred))
print(accuracy_score(y_test, test_pred))

#Tree Structure
from sklearn import tree
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (25,20))
tree.plot_tree(dt,
               feature_names= dataset.feature_names,
               class_names= dataset.target_names,filled = True
              )
plt.show()
