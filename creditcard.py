
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cc = pd.read_csv("creditcard.csv")

cc.head()

cc.drop(['Unnamed: 0'], axis=1, inplace=True)

cc.head(4)

import seaborn as sns
sns.boxplot(x='age', y='card', data =cc)
sns.boxplot(x='income', y='card', data =cc)
sns.boxplot(x='share', y='card', data =cc)
sns.boxplot(x='months', y='card', data =cc)
sns.boxplot(x='active', y='card', data =cc)


#Checking the independence between the independent variables
sns.heatmap(cc.corr())

#Data preprocessing and encoding
from sklearn import preprocessing
encoding = preprocessing.LabelEncoder()

cc['card']=encoding.fit_transform(cc['card'])
cc['owner']=encoding.fit_transform(cc['owner'])
cc['selfemp']=encoding.fit_transform(cc['selfemp'])
cc['card'].unique()
cc['owner'].unique()
cc['selfemp'].unique()
type('card')

# Converting datatype to 'category'
cc['card'] = cc['card'].astype('category')
cc['owner'] = cc['owner'].astype('category')
cc['selfemp'] = cc['selfemp'].astype('category')
\
cc.describe()
cc.mean()
cc.apply(lambda x:x.mean())
cc.isna().sum()

# Splitting data into train and test data
from sklearn.model_selection import train_test_split
x = cc.drop('card', axis =1) 
y = cc['card']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
# Building the model
model = LogisticRegression()
model.fit(x_train, y_train)

#Predicting
y_pred = model.predict(x_test)

# Test results and confusion matrix 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)

# Computing accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the model on logistic regression :", accuracy)
# 97.97%

# Computing precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



