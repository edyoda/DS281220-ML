# -*- coding: utf-8 -*-

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


estima = [
    ('rf', RandomForestClassifier(n_estimators=20)),
    ('svc', SVC(kernel='rbf', probability=True)),
    ('knc', KNeighborsClassifier()),
    ('abc', AdaBoostClassifier(base_estimator= DecisionTreeClassifier(), n_estimators=20)),
    ('lr', LogisticRegression())
    ]

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state=0)

vc = VotingClassifier(estimators = estima, voting = 'hard')
vc.fit(X_train, y_train)
for est, name in zip(vc.estimators_, vc.estimators):
    print(name[0], est.score(X_test, y_test))
print(vc.score(X_test, y_test))

#Soft Voting
vc_s = VotingClassifier(estimators = estima, voting='soft', weights = [3, 5, 1, 2, 4])
vc_s.fit(X_train, y_train)

print(vc_s.score(X_test, y_test))
