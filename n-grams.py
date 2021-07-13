# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:49:55 2021

@author: RISHBANS
"""

from nltk.collocations import BigramCollocationFinder

word_list = ['I', 'believe', 'would', 'help', 'reader', 'understand', \
             'tokenization', 'works', 'I', 'believe','As', 'well', 'realize', 'importance', 'text']
    
finde = BigramCollocationFinder.from_words(word_list)    
print(finde.ngram_fd.items())