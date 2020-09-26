#Importing basic modules
import numpy as np
import pandas as pd

# Importing necessary modules for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #For Activation,Layer,Lambda
from sklearn.model_selection import train_test_split

ff = pd.read_csv("forestfires.csv")

ff.head(3)
ff.columns
ff.shape

ff.isnull().sum() # No missing values 

#Assigning values to categorical variable 
# 'small' = 1 and 'large' = 0
ff.loc[ff.size_category=="small","size_category"] = 1
ff.loc[ff.size_category=="large","size_category"] = 0

# Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# Assigning dummy values 
ff['month'] = label_encoder.fit_transform(ff['month'])
ff['day'] = label_encoder.fit_transform(ff['day'])
# Converting to category type
ff['month'] = ff['month'].astype('category')
ff['day'] = ff['day'].astype('category')

ff.size_category.value_counts()
ff.size_category.value_counts().plot(kind="bar")

#Splitting data into train and test data
train,test = train_test_split(ff,test_size = 0.3,random_state=42)

trainX = train.drop(["size_category"],axis=1)
trainY = train["size_category"]
testX = test.drop(["size_category"],axis=1)
testY = test["size_category"]

test.size_category.value_counts()
train.size_category.value_counts()

# Preparing a function to define the structure of ANN network 
# Number hidden neurons & Hidden Layers 
# Activation function 
# Optimizer - similar to that of gradient decent 
# loss - loss function 
# rmsprop - > Root mean sqaure prop 

def prep_model(hidden_dim):
    model = Sequential() # initialize 
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    # To define the dimensions for the output layer
    # activation - sigmoid 
    model.add(Dense(hidden_dim[-1],kernel_initializer="normal",activation="sigmoid"))
    # loss function -> loss parameter
    # algorithm to update the weights - optimizer parameter
    # accuracy - metric to display for 1 epoch
    model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])
    return model    


# giving input as list format which is referring to
# number of input features - 30
# number of hidden neurons in each hidden layer - 4  layers
# 50- hidden neurons - 1st hidden layer
# Number of dimensions for output layer  - last 
first_model = prep_model([30,50,40,20,1])

# Fitting ANN model with epochs = 100 
first_model.fit(np.array(trainX),np.array(trainY),epochs=100)

# Predicting the probability values for each record for train data
pred_train = first_model.predict(np.array(trainX))

# pd.Series - > convert list format Pandas Series data structure
pred_train = pd.Series([i[0] for i in pred_train])

forestfire_size_class = ["small","large"]
# Converting to series to add them as columns into data frame
pred_train_class = pd.Series(["large"]*361)
pred_train_class[[i>0.5 for i in pred_train]] = "small"

from sklearn.metrics import confusion_matrix
train["original_class"] = "large"
train.loc[train.size_category==1,"original_class"] = "small"
train.original_class.value_counts()

# Two way table format 
confusion_matrix(pred_train_class,train.original_class)

# Calculating the accuracy using mean function from numpy 
# Resetting the index values of train data as the index values are random numbers
np.mean(pred_train_class==pd.Series(train.original_class).reset_index(drop=True))
#99.44%

# 2 way table 
pd.crosstab(pred_train_class,pd.Series(train.original_class).reset_index(drop=True))

# Predicting for test data 
pred_test = first_model.predict(np.array(testX))

pred_test = pd.Series([i[0] for i in pred_test])
pred_test_class = pd.Series(["large"]*156)

pred_test_class[[i>0.5 for i in pred_test]] = "small"
test["original_class"] = "large"
test.loc[test.size_category==1,"original_class"] = "small"
test.original_class.value_counts()
temp = pd.Series(test.original_class).reset_index(drop=True)
np.mean(pred_test_class==pd.Series(test.original_class).reset_index(drop=True)) 
#96.15%
len(pred_test_class==pd.Series(test.original_class).reset_index(drop=True))
confusion_matrix(pred_test_class,temp)
pd.crosstab(pred_test_class,test.original_class)

# Plot to show the count of each category with respect to other category 
test.original_class.value_counts().plot(kind="bar")
pred_test_class.value_counts().plot(kind="bar")