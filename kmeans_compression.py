# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 20:25:58 2021

@author: RISHBANS
"""

from skimage.io import imread, imshow
imshow('car.jpg')

img = imread('car.jpg')
print(img.shape)

img = img/255
print(img)
print(img.shape)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters = 6, init = 'k-means++')
img_new = img.reshape(1280*1920, 3)

print(img_new.shape)
kmeans.fit(img_new)
print(kmeans.cluster_centers_)
plt.imshow(kmeans.cluster_centers_)


img_com = kmeans.cluster_centers_[[kmeans.labels_]]
imshow(img_com.reshape(1280,1920,3))

