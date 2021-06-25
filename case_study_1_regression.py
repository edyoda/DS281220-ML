# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 18:59:30 2021

@author: RISHBANS
"""

from sklearn.datasets import load_diabetes
import pandas as pd
data_diab = load_diabetes()
features = data_diab.data
target = data_diab.target

#store data into a dataframe
house_data = pd.DataFrame( features, columns = data_diab.feature_names )
house_data['Progress'] = target

#split into X and y - feature/target variables
X = house_data.iloc[:, :-1].values
y = house_data.iloc[:, -1].values

#split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Apply the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Calculate Score
print(model.score(X_test, y_test))



from sklearn.ensemble import GradientBoostingRegressor
grad = GradientBoostingRegressor()
grad.fit(X_train, y_train)
print(grad.score(X_test, y_test))







