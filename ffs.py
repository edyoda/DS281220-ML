# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("pima-data.csv")

corr = df.corr()
del df['skin']

#Data Molding
diab_map = {True: 1, False:0}
df['diabetes'] = df['diabetes'].map(diab_map)

X = df.iloc[:, 0:8]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
#Forward feature Selection
sfs_c = sfs(nb_model, k_features = 6, forward = True, verbose = 3, scoring = 'accuracy', cv = 4)

sfs_c.fit(X_train, y_train)
feature_select = list(sfs_c.k_feature_idx_)
print(feature_select)


nb_model.fit(X_train[:, feature_select], y_train)

y_pred = nb_model.predict(X_test[:, feature_select])

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


