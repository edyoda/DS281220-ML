# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:07:29 2021

@author: RISHBANS
"""

from nltk.tokenize import sent_tokenize, word_tokenize

text = "I believe this would help the reader understand how tokenization \
        works. As well realize its importance."
        
sents = (sent_tokenize(text))
print(sents)
words = (word_tokenize(text))
print(words)

words_sent = [word_tokenize(sent) for sent in sents]
print(words_sent)