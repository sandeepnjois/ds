#Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#Importing dataset
cd = pd.read_csv("Company_Data.csv")
cd
cd.head()
colnames = list(cd.columns)
print(colnames)

#EDA
cd.isnull().sum()
# No N.A values

#Sales
cd['Sales'].mean() 
cd['Sales'].median()
cd['Sales'].mode()
cd['Sales'].var()
cd['Sales'].std()

cd['Sales'].skew()
cd['Sales'].kurt()
#skewness = slight right skewed
#kurtosis = slight flat curve
plt.boxplot(cd['Sales'],1,'rs',0)
# 2 outliers found

#CompPrice
cd['CompPrice'].mean() 
cd['CompPrice'].median()
cd['CompPrice'].mode()
cd['CompPrice'].var()
cd['CompPrice'].std()

cd['CompPrice'].skew()
cd['CompPrice'].kurt()
#skewness = slight left skewed
#kurtosis = slight thin curve
plt.boxplot(cd['CompPrice'],1,'rs',0)
# 2 outliers found

#Income
cd['Income'].mean() 
cd['Income'].median()
cd['Income'].mode()
cd['Income'].var()
cd['Income'].std()

cd['Income'].skew()
cd['Income'].kurt()
#skewness = slight right skewed
#kurtosis = slight flat curve
plt.boxplot(cd['Income'],1,'rs',0)
# no outliers found

#Advertising
cd['Advertising'].mean() 
cd['Advertising'].median()
cd['Advertising'].mode()
cd['Advertising'].var()
cd['Advertising'].std()

cd['Advertising'].skew()
cd['Advertising'].kurt()
#skewness = moderately right skewed
#kurtosis = moderately flat curve
plt.boxplot(cd['Advertising'],1,'rs',0)
# no outliers found

#Population
cd['Population'].mean() 
cd['Population'].median()
cd['Population'].mode()
cd['Population'].var()
cd['Population'].std()

cd['Population'].skew()
cd['Population'].kurt()
#skewness = slight left skewed
#kurtosis = flat curve
plt.boxplot(cd['Population'],1,'rs',0)
# no outliers found

#Price
cd['Price'].mean() 
cd['Price'].median()
cd['Price'].mode()
cd['Price'].var()
cd['Price'].std()

cd['Price'].skew()
cd['Price'].kurt()
#skewness = slight left skewed
#kurtosis = moderately flat curve
plt.boxplot(cd['Price'],1,'rs',0)
# 5 outliers found

#Age
cd['Age'].mean() 
cd['Age'].median()
cd['Age'].mode()
cd['Age'].var()
cd['Age'].std()

cd['Age'].skew()
cd['Age'].kurt()
#skewness = slight left skewed
#kurtosis = flat curve
plt.boxplot(cd['Age'],1,'rs',0)
# no outliers found

#Education
cd['Education'].mean() 
cd['Education'].median()
cd['Education'].mode()
cd['Education'].var()
cd['Education'].std()

cd['Education'].skew()
cd['Education'].kurt()
#skewness = slight right skewed
#kurtosis = flat curve
plt.boxplot(cd['Education'],1,'rs',0)
# no outliers found


cd['Sales'].unique() 
#'Sales' is continuous, it is needed to convert it to category type
cd["Sales"].min()
cd["Sales"].max()

# Data preprocessing
from sklearn.preprocessing import LabelEncoder

# Converting continuous 'Sales' variable into categorical using 'cut' function
label_encoder = LabelEncoder()
n_bins = 2 #Creating 2 bins
sales_y = label_encoder.fit_transform(pd.cut(cd['Sales'], n_bins, retbins=True)[0])
print(sales_y)

cd = cd.drop("Sales", axis =1) # Dropping continuous 'sales' column
cd['sales_y'] = sales_y # Adding categorical 'sales' column

# Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# Assigning dummy values 
cd['Urban'] = label_encoder.fit_transform(cd['Urban'])
cd['US'] = label_encoder.fit_transform(cd['US'])
cd['ShelveLoc'] = label_encoder.fit_transform(cd['ShelveLoc'])

# Converting to category type
cd['Urban'] = cd['Urban'].astype('category')
cd['US'] = cd['US'].astype('category')
cd['ShelveLoc'] = cd['ShelveLoc'].astype('category')

# Splitting the data into training and testing data set
from sklearn.model_selection import train_test_split
x = cd.drop('sales_y', axis=1)  #Predictors
y = cd['sales_y'] #Target variables

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
#Accuracy = 72.5%

# Visualizing DecisionTree
from sklearn import tree
predictors = cd.drop('sales_y', axis = 1)
target = cd['sales_y']
model = model.fit(predictors,target)

plt.figure(figsize=(30,45))
tree.plot_tree(model)
