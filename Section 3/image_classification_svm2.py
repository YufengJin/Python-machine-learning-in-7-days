#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:39:29 2020

@author: z0024094
"""

from sklearn.svm import SVC
from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

svmClassifierLinear = SVC(kernel='linear')

# we train the SVC model with a linear kernel
svmClassifierLinear.fit(x_train, y_train)

y_predictionsSVMLinear = svmClassifierLinear.predict(x_test)

scoreSVMLinear = svmClassifierLinear.score(x_test, y_predictionsSVMLinear)
print(scoreSVMLinear)