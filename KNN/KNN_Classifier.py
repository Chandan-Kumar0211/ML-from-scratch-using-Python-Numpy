#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def eucledian_dist(a,b):
    return np.sqrt(np.sum((a-b)**2))


# In[3]:


class KNNclassifier:
    def __init__(self,k=5):
        self.k = k
    
    def fit(self,X_train,y_train):
        # Since in KNN there is no training involved, so we can simply pass the X_train and y_train values for fitting the model
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self,X_test):
        predictions = []
        for x_i in X_test:
            
            # Step:1- Calculating the distance btw each X_test sample from whole X_train sample
            distance = []
            for x_j in self.X_train:
                dist = eucledian_dist(x_i,x_j)
                distance.append(dist)
            
            # Step:2- Finding K nearest neighbours and their labels
            k_indices = np.argsort(distance)[:self.k]
            
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Step:3- Now getting the most common labels for Classification i.e. label with majority vote
            labels_dict = {}
            for i in k_nearest_labels:
                if i not in labels_dict.keys():
                    labels_dict[i] = 1
                else:
                    labels_dict[i] = labels_dict[i] + 1
            
            values_list = list(labels_dict.values())
            sorted_list_indices = np.argsort(values_list)
            max_label_indices = sorted_list_indices[-1]
            
            keys_list = list(labels_dict)
            most_common_label = keys_list[max_label_indices]
            
            predictions.append(most_common_label) 
        
        return predictions
            
         
    

