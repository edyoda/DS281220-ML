# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:09:14 2021

@author: RISHBANS
"""

import pandas as pd

dataset = pd.read_csv("Apply_Job.csv")

X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = dt.predict(X_test)

print(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
#Define Variables
clf = dt
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
plt.title('Decision Tree')
plt.xlabel('Exp in Year')
plt.ylabel('Salary in Lakh')
plt.legend()    
plt.show()