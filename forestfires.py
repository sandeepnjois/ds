import pandas as pd
import numpy as np
import seaborn as sns

# Importing file
forestfires = pd.read_csv("forestfires.csv")
forestfires.head()
forestfires.describe()
colnames = forestfires.columns
print(colnames)


# Boxplot of size_category with different variables
sns.boxplot(x= "FFMC", y= "size_category", data=forestfires, palette = "hls")
sns.boxplot(x = "DMC", y= "size_category", data=forestfires, palette= "hls")
sns.boxplot(x = "DC", y= "size_category", data=forestfires, palette= "hls")
sns.boxplot(x = "ISI", y= "size_category", data=forestfires, palette= "hls")
sns.boxplot(x = "temp", y= "size_category", data=forestfires, palette= "hls")
sns.boxplot(x = "RH", y= "size_category", data=forestfires, palette= "hls")
sns.boxplot(x = "wind", y= "size_category", data=forestfires, palette= "hls")

forestfires['month'].unique()
forestfires['day'].unique()
forestfires['size_category'].unique()

# Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

forestfires['month']=label_encoder.fit_transform(forestfires['month'])
forestfires['day']=label_encoder.fit_transform(forestfires['day'])
forestfires['size_category']=label_encoder.fit_transform(forestfires['size_category'])

forestfires['month'].unique()
forestfires['day'].unique()
forestfires['size_category'].unique()

# Converting datatype to 'category'
forestfires['month']=forestfires['month'].astype('category')
forestfires['day']=forestfires['day'].astype('category')
forestfires['size_category']=forestfires['size_category'].astype('category')


# Importing 'Support vector machine' classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x = forestfires.drop('size_category', axis =1 )
y = forestfires['size_category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

x_train.head()
y_train.head()
x_test.head()
y_test.head()

# Creatting SVM classification object 
# Kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train, y_train)
pred_linear_test = model_linear.predict(x_test)

np.mean(pred_linear_test==y_test) #Accuracy = 96.79%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train, y_train)
pred_poly_test = model_poly.predict(x_test)

np.mean(pred_poly_test==y_test) #Accuracy = 75.64%

# Kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train, y_train)
pred_rbf_test = model_rbf.predict(x_test)

np.mean(pred_rbf_test==y_test) #Accuracy = 72.43%

# Kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(x_train, y_train)
pred_sigmoid_test = model_sigmoid.predict(x_test)

np.mean(pred_sigmoid_test==y_test) #Accuracy = 71.15%

# Linear kernel has the best accuracy
