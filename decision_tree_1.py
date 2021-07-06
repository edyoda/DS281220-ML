# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 19:27:38 2021

@author: RISHBANS
"""

from sklearn.datasets import make_blobs
X,y = make_blobs(n_features=2, n_samples=1000, cluster_std=0.8, centers=4, random_state=6)

import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.colorbar()


import pandas as pd
df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1]})
df['target'] = y

from sklearn.tree import DecisionTreeClassifier
import numpy as np

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(df[['X1', 'X2']], df.target)

clf = dt
h = 0.1
X_plot, z_plot = X, y
#Standard Template
x_min, x_max = X_plot[:, 0].min() -1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() -1, X_plot[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

outcome = dt.predict(np.c_[xx.ravel(), yy.ravel()])
print(outcome)


plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.colorbar()
plt.scatter(xx.ravel(), yy.ravel(), c = outcome, s=0.5, alpha = 0.2)


#KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
knc.fit(df[['X1', 'X2']], df.target)

outcome = knc.predict(np.c_[xx.ravel(), yy.ravel()])
plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0] - KNei')
plt.ylabel('X[1]')
plt.colorbar()
plt.scatter(xx.ravel(), yy.ravel(), c = outcome, s=0.5, alpha = 0.2)



#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(df[['X1', 'X2']], df.target)

outcome = lr.predict(np.c_[xx.ravel(), yy.ravel()])
plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0] - Logistic Regression')
plt.ylabel('X[1]')
plt.colorbar()
plt.scatter(xx.ravel(), yy.ravel(), c = outcome, s=0.5, alpha = 0.2)











