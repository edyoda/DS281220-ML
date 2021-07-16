# -*- coding: utf-8 -*-
import pandas as pd
dataset = pd.read_csv("Job_Exp.csv")

X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X, y)

y_pred = dt.predict([[27]])
print(y_pred)

#Visualization
import matplotlib.pyplot as plt
import numpy as np
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, dt.predict(X_grid), color = 'red')
plt.ylabel('Getting Job Chance in %')
plt.xlabel('years of exp')