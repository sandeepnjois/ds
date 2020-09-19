#Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


#Importing dataset
f = pd.read_csv("Fraud_check.csv")
f.head()
colnames = list(f.columns)
print(colnames)

f.isnull().sum()
# No N.A values

#Taxable.Income
f['Taxable.Income'].mean() 
f['Taxable.Income'].median()
f['Taxable.Income'].mode()
f['Taxable.Income'].var()
f['Taxable.Income'].std()

f['Taxable.Income'].skew()
f['Taxable.Income'].kurt()
#skewness = slight right skewed
#kurtosis = flat curve
plt.boxplot(f['Taxable.Income'],1,'rs',0)
#no outliers found

#City.Population
f['City.Population'].mean() 
f['City.Population'].median()
f['City.Population'].mode()
f['City.Population'].var()
f['City.Population'].std()

f['City.Population'].skew()
f['City.Population'].kurt()
#skewness = slight right skewed
#kurtosis = flat curve
plt.boxplot(f['City.Population'],1,'rs',0)
#no outliers found

#Work.Experience
f['Work.Experience'].mean() 
f['Work.Experience'].median()
f['Work.Experience'].mode()
f['Work.Experience'].var()
f['Work.Experience'].std()

f['Work.Experience'].skew()
f['Work.Experience'].kurt()
#skewness = slight right skewed
#kurtosis = flat curve
plt.boxplot(f['Work.Experience'],1,'rs',0)
#no outliers found

f['Taxable.Income'].unique() 
# 'Taxable.Income' is continuous, it is needed to convert it to category type
f["Taxable.Income"].min()
f["Taxable.Income"].max()

# Data preprocessing
from sklearn.preprocessing import LabelEncoder 

# Adding new categorical column 'group'
# 'group' ==> Risky <=30,000/- , Good >30,000/- ( of 'Taxable.Income')
f['group'] = pd.cut(f['Taxable.Income'],
                     bins=[0, 30000, 100000],
                     labels=["Risky", "Good"])

f = f.drop("Taxable.Income", axis = 1) # Dropping 'Taxable.Income' continuous columns

# Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# Assigning dummy values 
f['Undergrad'] = label_encoder.fit_transform(f['Undergrad'])
f['Marital.Status'] = label_encoder.fit_transform(f['Marital.Status'])
f['Urban'] = label_encoder.fit_transform(f['Urban'])

# Converting to category type
f['Undergrad'] = f['Undergrad'].astype('category')
f['Marital.Status'] = f['Marital.Status'].astype('category')
f['Urban'] = f['Urban'].astype('category')

# Splitting the data into training and testing data set
from sklearn.model_selection import train_test_split
x = f.drop('group', axis=1) #Predictors
y = f['group'] #Target variables

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.2, random_state=0)

train_x.head()
test_x.head()
train_y.head()
test_y.head()

# Importing Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="entropy")
dt_model = model.fit(train_x,train_y)

#Predict the response for test dataset
pred_y = dt_model.predict(test_x)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, pred_y))
#Accuracy = 68.33%

# Visualizing DecisionTree
from sklearn import tree
predictors = f.drop('group', axis = 1)
target = f['group']
model = model.fit(predictors,target)

plt.figure(figsize=(30,45))
tree.plot_tree(model)
