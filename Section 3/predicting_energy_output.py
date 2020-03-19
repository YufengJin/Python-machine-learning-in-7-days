#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:10:13 2020

@author: z0024094
"""

import numpy
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
filename = "power.csv"
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")


X=data[:,0:4]
y=data[:,4]

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42)   
print(X_trn.shape)
print(y_trn.shape)
print(X_tst.shape)
print(y_tst.shape)


plt.scatter(X_trn[:,0], y_trn,  color='red')
plt.xlabel('Ambient Temperature')
plt.ylabel('energy output')
plt.show()


# Plot outputs Exhaust Vacuum vs energy output
plt.scatter(X_trn[:,1], y_trn,  color='red')
plt.xlabel('Exhaust Vacuum')
plt.ylabel('energy output')
plt.show()

# Plot outputs Ambient Pressure (AP) vs energy output
plt.scatter(X_trn[:,2], y_trn,  color='red')
plt.xlabel('Ambient Pressure')
plt.ylabel('energy output')
plt.show()

# Plot outputs Relative Humidity vs energy output
plt.scatter(X_trn[:,3], y_trn,  color='red')
plt.xlabel('Relative Humidity')
plt.ylabel('energy output')
plt.show()


regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_trn, y_trn)

# Make predictions using the testing set
y_pred = regr.predict(X_tst)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_tst, y_pred))

plt.scatter(X_tst[:,0],y_tst,color='blue')
plt.xlabel('test data of x[:,0]')
plt.ylabel('y test data')
plt.show()