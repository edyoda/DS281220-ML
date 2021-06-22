# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:59:22 2021

@author: RISHBANS
"""

import pandas as pd

df = pd.read_csv("pima-data.csv")

df.isnull().values.any()

corr = df.corr()

del df['skin']

# Data Moulding
diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

feature_col_names =['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi','diab_pred', 'age' ]
target_var = ['diabetes']

X = df[feature_col_names].values
y = df[target_var].values

#Splitting data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Imputing
from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values=0, strategy='mean')

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train.ravel())

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classifier.score(X_test, y_test))






















