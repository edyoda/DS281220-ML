# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:40:17 2021

@author: RISHBANS
"""

import pandas as pd
df = pd.read_csv("exams.csv")

#1. Data Standardisation
from sklearn.preprocessing import scale

df[['math score']] = scale(df[['math score']])
df[['reading score']] = scale(df[['reading score']])
df[['writing score']] = scale(df[['writing score']])

#label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

#One-hot Encoding
df = pd.get_dummies(df, columns = ['race/ethnicity', 'parental level of education', 'lunch', 
                                   'test preparation course'])

X = df.iloc[:, 1:19].values
y = df.iloc[:, 0].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


print(classifier.score(X_test, y_test))





















