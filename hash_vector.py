# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:38:17 2021

@author: RISHBANS
"""
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd

corpus = [
     'This is the first first document from heaven',
     'but the second document is from mars',
     'And this is the third one from nowhere',
     'Is this the first document from nowhere?',
]
df = pd.DataFrame({'text': corpus})

hash_v = HashingVectorizer(n_features = 14, norm=None, alternate_sign=False)
X = hash_v.fit_transform(df.text).toarray()
print(X)
