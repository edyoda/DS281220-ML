# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:31:41 2021

@author: RISHBANS
"""

import pandas as pd
df = pd.read_csv("pima-data.csv")

cor = df.corr()
del df['skin']

dia_dict = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(dia_dict)

X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values = 0, strategy = 'mean')

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))