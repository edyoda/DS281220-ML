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

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explain_var = pca.explained_variance_ratio_
print(explain_var)



from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_model_pred = nb_model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, nb_model_pred))









