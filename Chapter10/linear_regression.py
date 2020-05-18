# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:07:18 2020

@author: DELL
"""
import numpy as np


class l_reg(object):
    
    def __init__(self, eta=0.001, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        self.w_per_epoch = np.zeros((self.n_iter, self.w_.shape[0]))
        
        for i in range(self.n_iter):
            output = self.predict(X)
            error = (y - output)
            #update weights
            self.w_[1:] += self.eta * (X.T.dot(error))
            self.w_[0] += (error).sum() * self.eta
            
            self.w_per_epoch[i,:] = self.w_
            
            cost = (error**2).sum() / 2
            
            self.cost_.append(cost)
        
    def predict(self, X):
        eq = np.dot(X, self.w_[1:]) + self.w_[0]
        return eq


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', 
                 header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 
              'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

X = df[['RM']].values
y = df['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = l_reg()
lr.fit(X_std, y_std)


plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % \
      sc_y.inverse_transform(price_std)) #10840
