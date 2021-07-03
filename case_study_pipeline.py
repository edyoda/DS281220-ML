# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:58:28 2021

@author: RISHBANS
"""

#1. Import dataset
import pandas as pd
dataset = pd.read_csv("HR.csv")
#2. Store values in X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values

#3. Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#4. StandardScalar
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#5. LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))


#6. GridSearch - logistic regression - C, max_iter 
from sklearn.model_selection import GridSearchCV
lr_g = LogisticRegression(penalty='l2')
max_iter = [100, 120, 140, 160]
C = [0.5, 0.7, 1, 2]
param_grid = dict(max_iter=max_iter, C = C)
grid = GridSearchCV(estimator=lr_g, param_grid=param_grid, cv = 4)
grid_result = grid.fit(X_train, y_train)
print(grid_result.best_score_)
print(grid_result.best_params_)

#7. Pipeline - StandardScalar, Logistic Regression
from sklearn.pipeline import make_pipeline
hr_pipeline = make_pipeline(StandardScaler(), LogisticRegression())

hr_pipeline.fit(X_train, y_train)
y_pred = hr_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#8. Pipeline - StandardScalar, SelectKBest, Logistic Regression
from sklearn.feature_selection import SelectKBest, f_classif
k_pipeline = make_pipeline(StandardScaler(), SelectKBest(k=4, score_func=f_classif),
                           LogisticRegression(C=0.7, max_iter=100))
print(k_pipeline)

k_pipeline.fit(X_train, y_train)
y_pred = k_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

params = {'selectkbest__k' : [3, 4, 5, 6],
          'logisticregression__C': [0.5, 0.7, 0.9, 1, 2],
          'logisticregression__max_iter': [100, 120, 140, 160]
          }

gs = GridSearchCV(k_pipeline, param_grid= params, cv = 4)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)























