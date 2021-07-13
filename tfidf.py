# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:47:30 2021

@author: RISHBANS
"""

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
     'This is the first first document from heaven',
     'but the second document is from mars',
     'And this is the third one from nowhere',
     'Is this the first document from nowhere?',
]

vector = TfidfVectorizer()
vector.fit(corpus)
print(vector.vocabulary_)
print(vector.idf_)