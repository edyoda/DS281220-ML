# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 20:11:21 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, 2:5].values

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters = 6, init = 'k-means++', random_state = 4)
k_means.fit(X)
print(k_means.labels_)


wcss = []
for k in range(1, 15):
    k_means = KMeans(n_clusters = k, init = 'k-means++', random_state = 4)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS Score')    
plt.show()