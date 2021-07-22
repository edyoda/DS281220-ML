# -*- coding: utf-8 -*-


from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

adaboost = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100), n_estimators=80)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

adaboost.fit(X_train, y_train)
print(adaboost.score(X_test, y_test))

rf = RandomForestClassifier(n_estimators=150)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))


from sklearn.linear_model import LogisticRegression
ada_lr = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=100), n_estimators=50)
ada_lr.fit(X_train, y_train)
print(ada_lr.score(X_test, y_test))

lr = LogisticRegression(max_iter=150)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))