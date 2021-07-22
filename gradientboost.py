# -*- coding: utf-8 -*-

import pandas as pd

dataset = pd.read_csv("BankNote_Authentication.csv")

X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=150, max_depth = 50, learning_rate = 0.01)

gb.fit(X_train, y_train)

print(gb.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
n_estimators = [100, 110, 140]
max_depth = [15, 25, 35]
learning_rate = [0.01, 0.5, 0.1]
param_grid = dict(n_estimators=n_estimators, max_depth = max_depth, learning_rate= learning_rate)

gb_gs = GradientBoostingClassifier()
grid = GridSearchCV(estimator=gb_gs, param_grid= param_grid, cv = 4)

grid_result = grid.fit(X_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)
