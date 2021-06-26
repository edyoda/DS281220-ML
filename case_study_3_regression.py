# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:12:13 2021

@author: RISHBANS
"""
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

# Store in DataFrame
dataset = pd.DataFrame(boston.data, columns = boston.feature_names)
dataset['price'] = boston.target
corr = dataset.corr()

del dataset['TAX']
# Split into X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Split into Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Apply the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#Calculate the score
print(lr.score(X_test, y_test))
