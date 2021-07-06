# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:06:40 2021

@author: RISHBANS
"""

import pandas as pd
tennis_data = pd.read_csv("tennis.csv")

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()

X = tennis_data.drop(columns=['play'])
y = tennis_data.play

X = oe.fit_transform(X)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X, y)

print(dt.predict([[2,1,0,0]]))






