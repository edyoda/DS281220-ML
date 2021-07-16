# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Tweets.csv")

print(dataset.isnull().values.any())
print(dataset.isnull().sum(axis=0))
dataset.fillna(0)

dataset.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
dataset_sentiment = dataset.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
dataset_sentiment.plot(kind='bar')


dataset.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')


#Data Cleaning
feature = dataset.text

import re
process_feature = []

for tweet in range(0, len(feature)):
    
    clean_tweet = re.sub(r'\s+[a-z]+://[a-z].[a-z]+/[a-zA-Z0-9]+', ' ', str(feature[tweet]))
    #filter special character
    clean_tweet = re.sub(r'\W', ' ', clean_tweet)
    
    #filtering out all single characters
    clean_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', clean_tweet)
    
    #filtering out single character from the start
    clean_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', clean_tweet)
    
    #replace multiple spaces with single space
    clean_tweet = re.sub(r'\s+', ' ', clean_tweet)
    
    #convert to lower case
    clean_tweet = clean_tweet.lower()
    
    process_feature.append(clean_tweet)
    
print(process_feature)    
    
    
    
    
    
    
    
    