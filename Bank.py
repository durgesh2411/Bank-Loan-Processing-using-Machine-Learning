import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('C:/Users/VISHU/3D Objects/Data Science in Python/Bank Loan Processing/Bank_loan.csv')
'''The Date Consists of 8,87,379 rows and 30 columns'''

datacopy = data.copy()
from sklearn.model_selection import train_test_split
data.info()
x = data[['annual_inc','income_cat', 'dti', 'interest_rate', 'total_pymnt', 'installment', 'recoveries', 'total_rec_prncp']]
y = data['loan_amount']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state = 0)
from sklearn.linear_model import LinearRegression

Regression = LinearRegression()
Regression.fit(xtrain, ytrain)
ypred = Regression.predict(xtest)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print("The Regression accuracy : ",Regression.score(xtest, ytest)*100)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
x_poly = poly.fit_transform(x)
lreg = LinearRegression()
lreg.fit(x_poly, y)
ypred = Regression.predict(xtest)
print("The Polynomial Regression accuracy is : ",Regression.score(xtest, ypred) * 100)

import pickle
pickle.dump(Regression, open("Bank.pkl", "wb"))