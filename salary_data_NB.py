#Importing modules
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Importing datasets
salary_train = pd.read_csv("SalaryData_Train.csv")
salary_test = pd.read_csv("SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

# Data preprocessing
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

#Naive bayes models
sgnb = GaussianNB()
smnb = MultinomialNB()

#GaussianNB model
#Building and predicting at the same time 

spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb) #Confusion matrix
pd.crosstab(testY.values.flatten(),spred_gnb) #Confusion matrix using crosstab
np.mean(spred_gnb==testY.values.flatten()) # 0.7949
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 79.49%
#OR
from sklearn import metrics
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(testY,spred_gnb)) #79.49%

# Multinomal model
# Building and predicting at the same time 

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb) #Confusion matrix
pd.crosstab(testY.values.flatten(),spred_mnb) #Confusion matrix using crosstab 
np.mean(spred_mnb==testY.values.flatten()) #0.7749
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75.93%
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(testY,spred_mnb)) #77.49%
