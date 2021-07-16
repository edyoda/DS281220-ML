# -*- coding: utf-8 -*-
# Import the libraries
import pandas as pd

#Import the csv file
dataset = pd.read_csv("emails.csv")

#save columns into X and y
X = dataset["text"]
y = dataset.spam
#Split in to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

#Apply countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
X_train = cv.fit_transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()

#Make diff model using: KNeighbors, Gaussian NB, Multinomial NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred))