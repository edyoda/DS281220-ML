# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:30:13 2021

@author: RISHBANS
"""

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

X = [[0,1], [1,1]]
Y = [[1,2], [2,2]]

print(euclidean_distances(X, Y))

print(euclidean_distances(X, X))

print(manhattan_distances(X, Y))

print(cosine_distances(X, Y))



