# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:03:23 2021

@author: RISHBANS
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

lr.score(X_test, y_test)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Define Variables
clf = lr
h = 0.1
X_plot, z_plot = X_train, y_train

#Standard Template
x_min, x_max = X_plot[:, 0].min() -1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() -1, X_plot[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z,
             alpha = 0.7, cmap = ListedColormap(('blue', 'red')))
 
for i , j in enumerate(np.unique(z_plot)):
    plt.scatter(X_plot[z_plot == j, 0], X_plot[z_plot == j, 1],
    c = ['blue', 'red'][i], cmap = ListedColormap(('blue', 'red')), label = j)
    
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Logistic Regression')
plt.xlabel('Feature - 1')
plt.ylabel('Feature - 2')
plt.legend()    
plt.show()

















