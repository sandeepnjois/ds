# Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

glass = pd.read_csv("glass.csv")

# Training and Test data using 
from sklearn.model_selection import train_test_split

x=glass.drop('Type', axis =1)
y=glass['Type']
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2, random_state=0)

train_x.head()
train_y.head()
test_x.head()
test_y.head()

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier as knc
# for 3 nearest neighbours 
neighbour = knc(n_neighbors=3)

# Fitting with training data 
neighbour.fit(train_x,train_y)

# train accuracy 
train_acc = np.mean(neighbour.predict(train_x)==train_y) 
train_acc # 83.62%
    
# test accuracy
test_acc = np.mean(neighbour.predict(test_x)==test_y) 
test_acc # 55.81%

# for 5 nearest neighbours
neighbour = knc(n_neighbors=5)

# fitting with training data
neighbour.fit(train_x,train_y)

# train accuracy 
train_acc = np.mean(neighbour.predict(train_x)==train_y)
train_acc # 77.19%

# test accuracy
test_acc = np.mean(neighbour.predict(test_x)==test_y)
test_acc # 58.14%

# Creating an empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours (odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neighbour = knc(n_neighbors=i)
    neighbour.fit(train_x,train_y)
    train_acc = np.mean(neighbour.predict(train_x)==train_y)
    test_acc = np.mean(neighbour.predict(test_x)==test_y)
    acc.append([train_acc,test_acc])

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

# Best accuracy at k = 19
# At k = 19, accuracy 
# train accuracy =  67.25% , test accuracy = 60.46%