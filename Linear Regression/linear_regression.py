#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class LinearRegression:
    
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        
        # Step:1- Initilizing the parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Step:2- Calculating gradients and applying gradient desent for updating weights
        for i in range(self.n_iter):
            y_pred = np.dot(X,self.weights) + self.bias
            
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)
            
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
        
    def predict(self,X):
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred

