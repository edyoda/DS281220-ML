# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:53:14 2021

@author: RISHBANS
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.data.shape)

#create dataframe
dataset = pd.DataFrame(cancer.data, columns = cancer.feature_names)
dataset['cancer'] = cancer.target

#split into X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#split into tran and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#apply GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#calcualte score
print(gnb.score(X_test, y_test))