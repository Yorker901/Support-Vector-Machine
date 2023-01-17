# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:34:12 2022

@author: Mohd Ariz Khan
"""
# Import the data
import pandas as pd
df = pd.read_csv("forestfires.csv")
df 

# Get information of the dataset
df.info()
df.shape
print('The shape of our data is:', df.shape)
df.isnull().any()

# Dropping the month and day columns
df.drop(["month","day"],axis=1,inplace =True)
df

#=============================================================================
#                      EDA (Exploratory Data Analysis)
#=============================================================================
# Pie Chart
import matplotlib.pyplot as plt
df['size_category'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

# Histogram
plt.hist(df['size_category'], bins=5)
plt.show()

# let's make scatter plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df, hue = 'size_category')

# split the variables as X and y
X = df.iloc[:,0:28]  # predictors variables
Y = df["size_category"] # target variable

# Normalising the data as there is scale difference
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_X = pd.DataFrame(MM.fit_transform(X))
MM_X

# Splitting Train and Test
from sklearn.model_selection._split import train_test_split
X_train,X_test,y_train,y_test = train_test_split(MM_X, Y, test_size=0.25, stratify = Y)

# Shape of the train and test data
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Model Fitting
from sklearn.svm import SVC
svc_class = SVC(kernel='linear')

svc_class.fit(X_train, y_train)
y_pred_train = svc_class.predict(X_train)
y_pred_test  = svc_class.predict(X_test)

from sklearn.metrics  import accuracy_score, confusion_matrix
print("Training Accuracy :",accuracy_score(y_train, y_pred_train)*100)
print("Testing Accuracy:",accuracy_score(y_test, y_pred_test)*100)
confusion_matrix(y_test, y_pred_test)

# From support Vector Classifier (Kernal = linear) 
# Training Accuracy : 74.41%
# Testing Accuracy: 76.15%
#-----------------------------------------------------------------------------

from sklearn.svm import SVC
svc_poly = SVC(kernel='poly', degree = 3)

svc_poly.fit(X_train, y_train)
y_pred_train = svc_poly.predict(X_train)
y_pred_test  = svc_poly.predict(X_test)

from sklearn.metrics  import accuracy_score, confusion_matrix
print("Training Accuracy :",accuracy_score(y_train, y_pred_train)*100)
print("Testing Accuracy:",accuracy_score(y_test, y_pred_test)*100)
confusion_matrix(y_test, y_pred_test)

# From support Vector Classifier (Kernal = poly) 
# Training Accuracy : 76.48%
# Testing Accuracy: 76.15%
#-----------------------------------------------------------------------------

from sklearn.svm import SVC
svc_RBF = SVC(kernel='rbf',gamma=4)

svc_RBF.fit(X_train, y_train)
y_pred_train = svc_RBF.predict(X_train)
y_pred_test  = svc_RBF.predict(X_test)

from sklearn.metrics  import accuracy_score
print("Training Accuracy :",accuracy_score(y_train, y_pred_train)*100)
print("Testing Accuracy:",accuracy_score(y_test, y_pred_test)*100)

# From support Vector Classifier (Kernal = rbf) 
# Training Accuracy : 79.32%
# Testing Accuracy: 74.61%
#----------------------------------------------------------------------


