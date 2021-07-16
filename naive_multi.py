# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:35:54 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("horror-train.csv")

print(dataset.info())
print(dataset.author.value_counts())

X = dataset.text
y = dataset.author

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
spam_fil = CountVectorizer(stop_words = 'english')
X_train = spam_fil.fit_transform(X_train).toarray()
X_test = spam_fil.transform(X_test).toarray()

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
print(mnb.score(X_test, y_test))

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
print(bnb.score(X_test, y_test))











