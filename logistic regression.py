# importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv(r'C:\Users\sai\Downloads\logit classification.csv')

# independent variable
x = dataset.iloc[:,[2,3]].values

# dependent variable
y = dataset.iloc[:, -1].values

# we are training, testing and splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# building logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l1', solver='liblinear', C=100.0)
classifier.fit(x_train, y_train)

# making predictions
y_pred = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy score
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# bias - training accuracy
bias = classifier.score(x_train, y_train)
print(bias)

# variance - testing accuray
variance = classifier.score(x_test, y_test)
print(variance)

#-------------------------FUTURE PREDICTION-----------------------------------
dataset1 = pd.read_csv(r'C:\Users\sai\Downloads\Future prediction1.csv')

d2 = dataset1.copy()

# extract the relevant columns for prediciton
#d2
dataset1 = dataset1.iloc[:, [2,3]].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
M = sc.fit_transform(dataset1)

# initiating the empty DataFrame for strong prediction
y_pred1 = pd.DataFrame()

# columns for the future prediction
d2["y_pred1"] = classifier.predict(M)

# saving the code
d2.to_csv('pred_model.csv')

# To get the path
import os
os.getcwd()



