# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:02:48 2021

@author: RISHBANS
"""
import pandas as pd
from nltk.corpus import stopwords
corpus = [
     'This is the first first document from heaven',
     'but the second document is from mars',
     'And this is the third one from nowhere',
     'Is this the first document from nowhere?',
]

df = pd.DataFrame({'text': corpus})

from sklearn.feature_extraction.text import CountVectorizer
count_v = CountVectorizer()
X = count_v.fit_transform(df.text).toarray()
print(X)
print(count_v.vocabulary_)

count_v_stop = CountVectorizer(stop_words=set(stopwords.words('english')))
X = count_v_stop.fit_transform(df.text).toarray()
print(X)
print(count_v_stop.vocabulary_)


