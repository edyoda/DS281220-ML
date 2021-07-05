# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 19:56:25 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("titanic3.csv")
dataset.drop(['name', 'ticket', 'cabin', 'boat', 'body'], axis = 1, inplace = True)
dataset.drop(['home.dest'], axis = 1, inplace = True)

X = dataset.drop(columns = ['survived'])
y= dataset.survived

print(dataset.isnull().sum(axis=0))
print(dataset.dtypes)

# Text = ['sex', 'embarked'] - OnehotEncoder, SimpleImputer
# ['age', 'fare'] - SimpleImputer, , StandardScaler

int_data = X.select_dtypes(include=['int64'])

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

pipeline_int = make_pipeline(SimpleImputer(missing_values = np.nan, strategy='mean'),
                             StandardScaler())
pipeline_obj = make_pipeline(SimpleImputer(missing_values = np.nan, strategy='most_frequent'),
                             OneHotEncoder())

#Combine above pipelines
preprocessor = make_column_transformer(
                (pipeline_int, ['age', 'fare']),
                (pipeline_obj, ['sex', 'embarked']),
                remainder = 'passthrough'
    )

master_pipeline = make_pipeline(preprocessor, SelectKBest(k=5, score_func=f_classif),
                                RandomForestClassifier(n_estimators = 100))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


master_pipeline.fit(X_train, y_train)
y_pred = master_pipeline.predict(X_test)

print(master_pipeline.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
params = {'selectkbest__k' : [3, 5, 6, 7],
          'randomforestclassifier__n_estimators': [20, 100, 120, 140]}
gs = GridSearchCV(master_pipeline, param_grid=params, cv=4)
gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)
print(gs.score(X_test, y_test))



















