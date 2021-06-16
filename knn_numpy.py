# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:44:31 2021

@author: RISHBANS
"""

#Import the dataset
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['type'] = iris.target

class MyKNN:
#1. select value of k
    def __init__(self, k):
        self.k = k
    
    def my_fit(self, feature_data, target_data):
        self.feature_data = np.array(feature_data)
        self.target_data = np.array(target_data)
        

#2. calculate_euclidiean_distance
    def calculate_euclidiean_distance(self, one_data):
        distances = np.sqrt(np.sum(np.square(self.feature_data - one_data), axis = 1))
        print(distances)
        return distances


#3. find_k_neighbors
    def find_k_neighbors(self, one_data):
        res = self.calculate_euclidiean_distance(one_data)
        return res.argsort()[:self.k]    

#4. find_k_neighbors_class
    def find_k_neighbors_class(self, one_data):
        index_of_neighbors = self.find_k_neighbors(one_data)
        print(index_of_neighbors)
        return self.target_data[index_of_neighbors]

#5. my_predict
    def my_predict(self, one_data):
        classes = self.find_k_neighbors_class(one_data)
        print(classes)
        return np.bincount(classes).argmax()

model = MyKNN(5)
feature_data = df.drop(columns=['type'], axis = 1)
target_data = df.type
model.my_fit(feature_data, target_data)

one_data = [1,2,3,4]
#model.find_k_neighbors_class(one_data)
print(model.my_predict(one_data))




