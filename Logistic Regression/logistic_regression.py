#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class LogisticRegression:
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
        
        # Calculating the gradients and applying gradient desent for updating parameters
        for _ in range(self.n_iter):
            y_pred = 1/(1+np.exp(-(np.dot(X,self.weights)+self.bias)))
            
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)
            
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
    
    def predict(self,X):
        y_pred = 1/(1+np.exp(-(np.dot(X,self.weights)+self.bias)))
        class_pred = []
        for i in range(len(y_pred)):
            if y_pred[i] <= 0.5:
                class_pred.append(0)
            else:
                class_pred.append(1)
        return class_pred 
        

