import pandas as pd
import numpy as np
import seaborn as sns

# Importing file
sd = pd.read_csv("salarydata.csv")
sd.head()
sd.describe()
sd.columns

sd['workclass'].unique()
sd['education'].unique()
sd['maritalstatus'].unique()
sd['occupation'].unique()
sd['relationship'].unique()
sd['race'].unique()
sd['sex'].unique()
sd['native'].unique()

# Boxplot of size_category with different variables
sns.boxplot(x= "age", y= "Salary", data=sd, palette = "hls")
sns.boxplot(x = "capitalgain", y= "Salary", data=sd, palette= "hls")
sns.boxplot(x = "educationno", y= "Salary", data=sd, palette= "hls")
sns.boxplot(x = "hoursperweek", y= "Salary", data=sd, palette= "hls")

# Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

sd['Salary']=label_encoder.fit_transform(sd['Salary'])
sd['workclass']=label_encoder.fit_transform(sd['workclass'])
sd['education']=label_encoder.fit_transform(sd['education'])
sd['maritalstatus']=label_encoder.fit_transform(sd['maritalstatus'])
sd['occupation']=label_encoder.fit_transform(sd['occupation'])
sd['relationship']=label_encoder.fit_transform(sd['relationship'])
sd['race']=label_encoder.fit_transform(sd['race'])
sd['sex']=label_encoder.fit_transform(sd['sex'])
sd['native']=label_encoder.fit_transform(sd['native'])

sd['Salary'].unique()
sd['workclass'].unique()
sd['education'].unique()
sd['maritalstatus'].unique()
sd['occupation'].unique()
sd['relationship'].unique()
sd['race'].unique()
sd['sex'].unique()
sd['native'].unique()

# Converting datatype to 'category'
sd['Salary']=sd['Salary'].astype('category')
sd['workclass']=sd['workclass'].astype('category')
sd['education']=sd['education'].astype('category')
sd['maritalstatus']=sd['maritalstatus'].astype('category')
sd['occupation']=sd['occupation'].astype('category')
sd['relationship']=sd['relationship'].astype('category')
sd['race']=sd['race'].astype('category')
sd['sex']=sd['sex'].astype('category')
sd['native']=sd['native'].astype('category')


sd.columns

# Importing 'Support vector machine' classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x = sd.drop('Salary', axis =1)
y = sd['Salary']

# Sample dataset
x = x.iloc[0:250,]
y = y.iloc[0:250,]

train_x,test_x,train_y,test_y = train_test_split(x,y, test_size = 0.3, random_state = 0)

train_x.head()
train_y.head()
test_x.head()
test_y.head()

type("Salary")

# Creatting SVM classification object 
# Kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_x, train_y)
pred_linear_test = model_linear.predict(test_x)

np.mean(pred_linear_test==test_y) # Accuracy = 84.44%


# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_x, train_y)
pred_poly_test = model_poly.predict(test_x)

np.mean(pred_poly_test==test_y) # Accuracy = 82.22%

# Kernel = rbf

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_x, train_y)
pred_rbf_test = model_rbf.predict(test_x)

np.mean(pred_rbf_test==test_y) # Accuracy = 82.22%

# Kernel = sigmoid

model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_x, train_y)
pred_sigmoid_test = model_sigmoid.predict(test_x)

np.mean(pred_sigmoid_test==test_y) # Accuracy = 84.44%

# Linear kernel and sigmoid kernel have the best accuracy


