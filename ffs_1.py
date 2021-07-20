# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("Largecap_Balancesheet.csv")

X = df.iloc[:, 0:4]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr = LinearRegression()

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
sfs_c = sfs(lr, k_features = 2, forward = False, verbose = 10, scoring = 'r2', cv = 4)

sfs_c.fit(X_train, y_train)
feature_select = list(sfs_c.k_feature_idx_)
print(feature_select)

lr.fit(X_train[:, feature_select], y_train)

y_pred = lr.predict(X_test[:, feature_select])
print(y_pred)
print(r2_score(y_test, y_pred))

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train[:, feature_select], y_train)

y_pred_dt = dt.predict(X_test[:, feature_select])
print(r2_score(y_test, y_pred_dt))