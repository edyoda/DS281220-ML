# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:08:49 2021

@author: RISHBANS
"""

from sklearn.datasets import load_digits
dataset = load_digits()
print(dataset.data.shape)
X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

trainX = ss.fit_transform(X_train)
testX = ss.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
class_rf = RandomForestClassifier()
class_rf.fit(trainX, y_train)

print(class_rf.predict(testX))

# Step 1 - Use Pipeline
from sklearn.pipeline import make_pipeline
digit_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())

digit_pipeline.fit(X_train, y_train)

y_pred = digit_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

print(digit_pipeline.steps[1][1].feature_importances_)


#Step 2 - Use Hyper paramater
from sklearn.feature_selection import SelectKBest, f_classif

digit_pipeline = make_pipeline(StandardScaler(), 
                               SelectKBest(k=10, score_func=f_classif), 
                               RandomForestClassifier(n_estimators = 100))

print(digit_pipeline)

digit_pipeline.fit(X_train, y_train)
y_pred = digit_pipeline.predict(X_test)

print(y_pred)
print(accuracy_score(y_test, y_pred))

#Step 3 - Estimating the best set of hyper parameters, GridSearch

from sklearn.model_selection import GridSearchCV

params = {'selectkbest__k': [20, 30, 50, 60, 64], 
          'randomforestclassifier__n_estimators': [150, 200, 250]}

gs = GridSearchCV(digit_pipeline, param_grid = params, cv = 4)
gs.fit(X_train, y_train)

print(gs.best_params_)
print(gs.best_score_)
y_pred = gs.predict(X_test)
print(y_pred)

print(gs.best_estimator_)































