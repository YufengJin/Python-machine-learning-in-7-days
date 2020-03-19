#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:59:12 2020

@author: z0024094
"""

import numpy
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

filename = "height-weight.csv"
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")

X=data[:,0]
y=data[:,1]

X_20 = X[:20]
y_20 = y[:20]


# Split the data into training/testing sets
X_train=X[:4500]
X_test=X[4500:]
# Split the targets into training/testing sets
y_train=y[:4500]
y_test=y[4500:]


X_train=X_train.reshape(-1, 1)
X_test=X_test.reshape(-1, 1)


regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)


y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)


plt.scatter(X_train, y_train,  color='black')
plt.scatter(X_test, y_pred, color='blue', linewidth=1)

#plt.xticks(())
#plt.yticks(())

plt.show()