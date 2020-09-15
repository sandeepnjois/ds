
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bank = pd.read_csv("bank-full.csv", sep=";")

bank.head()
colnames = bank.columns
print(colnames)



import seaborn as sns
sns.boxplot(x='age', y='y', data =bank)
sns.boxplot(x='balance', y='y', data =bank)

# Onjects in the variables 
bank['job'].unique()
bank['marital'].unique()
bank['education'].unique()
bank['default'].unique()
bank['housing'].unique()
bank['loan'].unique()
bank['contact'].unique()
bank['month'].unique()
bank['poutcome'].unique()
bank['y'].unique()


#Checking the independence between the independent variables
sns.heatmap(bank.corr())

#Data preprocessing and encoding
from sklearn import preprocessing
encoding = preprocessing.LabelEncoder()

bank['job']=encoding.fit_transform(bank['job'])
bank['marital']=encoding.fit_transform(bank['marital'])
bank['education']=encoding.fit_transform(bank['education'])
bank['default']=encoding.fit_transform(bank['default'])
bank['housing']=encoding.fit_transform(bank['housing'])
bank['loan']=encoding.fit_transform(bank['loan'])
bank['contact']=encoding.fit_transform(bank['contact'])
bank['month']=encoding.fit_transform(bank['month'])
bank['poutcome']=encoding.fit_transform(bank['poutcome'])
bank['y']=encoding.fit_transform(bank['y'])


bank['job'].unique()
bank['marital'].unique()
bank['education'].unique()
bank['default'].unique()
bank['housing'].unique()
bank['loan'].unique()
bank['contact'].unique()
bank['month'].unique()
bank['poutcome'].unique()
bank['y'].unique()


# Converting datatype to 'category'
bank['job'] = bank['job'].astype('category')
bank['marital'] = bank['marital'].astype('category')
bank['education'] = bank['education'].astype('category')
bank['default'] = bank['default'].astype('category')
bank['housing'] = bank['housing'].astype('category')
bank['loan'] = bank['loan'].astype('category')
bank['contact'] = bank['contact'].astype('category')
bank['month'] = bank['month'].astype('category')
bank['poutcome'] = bank['poutcome'].astype('category')
bank['y'] = bank['y'].astype('category')


bank.describe()
bank.mean()

# N.A values
bank.isna().sum()

# Splitting data into train and test data
from sklearn.model_selection import train_test_split
x = bank.drop('y', axis =1) 
y = bank['y']

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
confusion_matrix

# Computing accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the model on logistic regression :", accuracy)
# Accuracy = 88.41%

# Computing precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



