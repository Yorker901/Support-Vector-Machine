# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:51:56 2022

@author: Mohd Ariz Khan
"""

# Import the data
import pandas as pd
df_test = pd.read_csv("SalaryData_Test(1).csv")
df_test

df_train = pd.read_csv("SalaryData_Train(1).csv")
df_train

# Get information of the datasets
df_test.info()
df_test.shape
print('The shape of our data is:', df_test.shape)
df_test.isnull().any()

df_train.info()
df_train.shape
print('The shape of our data is:', df_train.shape)
df_train.isnull().any()

#=============================================================================
#                      EDA (Exploratory Data Analysis)
#=============================================================================
# Pie Chart
import matplotlib.pyplot as plt
df_test['Salary'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

# Histogram
plt.hist(df_test['Salary'], bins=5)
plt.show()

# let's make scatter plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df_test, hue = 'Salary')

# Data Transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_test['Salary'] = LE.fit_transform(df_test['Salary'])
df_train['Salary'] = LE.fit_transform(df_train['Salary'])

df_new_test = pd.get_dummies(df_test)
df_new_train = pd.get_dummies(df_train)

# Drop the variable and split the variable from the data
x_train = df_new_train.drop('Salary',axis=1)
y_train = df_new_train['Salary']
x_test = df_new_test.drop('Salary',axis=1)
y_test = df_new_test['Salary']

y_train  

# Model Fitting using SVC (Suport Vector Machine)
from sklearn.svm import SVC
svc = SVC()

svc.fit(x_train, y_train)
y_pred_test  = svc.predict(x_test)
y_pred_test

# Classsification and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix

# Confusion Matrix of data
print(confusion_matrix(y_pred_test,y_test))

# Classsification of data
print(classification_report(y_pred_test,y_test))

import numpy as np
np.mean(y_pred_test == y_test)*100

# prediction of salary = 79.64%


