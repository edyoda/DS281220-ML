# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:16:32 2021

@author: RISHBANS
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
text = "I believe this would help the reader understand how tokenization \
        works. As well realize its importance (text)."
        
custom_list = set(stopwords.words('english')+list(punctuation))

word_list = [word for word in word_tokenize(text) if word not in custom_list]
print(word_list)

