# -*- coding: utf-8 -*-
# Import the libraries
import pandas as pd

#Import the csv file
dataset = pd.read_csv("Sentiment.csv")

#Drop the class neutral
dataset = dataset.drop(dataset[dataset.sentiment == "Neutral"].index)

#save columns into X and y
sent_map = {"Positive": 1, "Negative": 0}
dataset["sentiment"] = dataset["sentiment"].map(sent_map)
X = dataset.text
y = dataset.sentiment

#Split in to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Apply TFIDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df = 0.7, min_df = 10, stop_words = 'english')
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()

# #Apply countvectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(stop_words='english')
# X_train = cv.fit_transform(X_train).toarray()
# X_test = cv.transform(X_test).toarray()

#Make diff model using: KNeighbors, Gaussian NB, Multinomial NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print(gnb.score(X_test, y_test))

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

print(mnb.score(X_test, y_test))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print(rf.score(X_test, y_test))








