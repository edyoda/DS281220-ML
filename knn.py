# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:52:50 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

import numpy as np
error_rate = []
for i in range(2, 25):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    i_pred = classifier.predict(X_test)
    error_rate.append(np.mean(i_pred != y_test))
    
print(error_rate)
    
    
    
    
    
    
    
    
    