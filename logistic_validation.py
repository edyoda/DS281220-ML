# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:31:16 2021

@author: RISHBANS
"""

import pandas as pd
df = pd.read_csv("pima-data.csv")

del df['skin']

diab_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diab_map)


X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=0.7, max_iter=250)
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits = 4)
result = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

print(result.mean())


from sklearn.model_selection import GridSearchCV

max_i = [100, 120, 140, 180, 250]
C_i = [0.7, 1, 2, 3, 4]

param_grid = dict(max_iter = max_i, C = C_i)
lr = LogisticRegression(penalty = 'l2')
grid = GridSearchCV(estimator = lr, param_grid = param_grid, cv=4)

grid_result = grid.fit(X_train, y_train)


print(grid_result.best_score_)

print(grid_result.best_params_)












