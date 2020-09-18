#Importing packages
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

# Kmeans on EastWestAirlines Data set 
ewa = pd.read_excel("EastWestAirlines.xlsx", sheet_name='data')
colnames = ewa.columns
print(colnames)

#EDA
ewa.describe()
ewa.isnull().sum() # No N.A values

#Business moments and visualizations
#Balance
ewa['Balance'].mean() 
ewa['Balance'].median()
ewa['Balance'].mode()
ewa['Balance'].var()
ewa['Balance'].std()

ewa['Balance'].skew()
ewa['Balance'].kurt()
#Skewness = Positive, right skewed
#Kurtosis = Thin and peaked curve
plt.boxplot(ewa['Balance'],1,'bs',0)

#Bonus_miles
ewa['Bonus_miles'].mean() 
ewa['Bonus_miles'].median()
ewa['Bonus_miles'].mode()
ewa['Bonus_miles'].var()
ewa['Bonus_miles'].std()

ewa['Bonus_miles'].skew()
ewa['Bonus_miles'].kurt()
#Skewness = Positive, right skewed
#Kurtosis = Thin and peaked curve
plt.boxplot(ewa['Bonus_miles'],1,'bs',0)

#Bonus_trans
ewa['Bonus_trans'].mean()
ewa['Bonus_trans'].median()
ewa['Bonus_trans'].mode()
ewa['Bonus_trans'].var()
ewa['Bonus_trans'].std()

ewa['Bonus_trans'].skew()
ewa['Bonus_trans'].kurt()
#Skewness = Positive, right skewed
#Kurtosis = Thin and peaked curve
plt.boxplot(ewa['Bonus_trans'],1,'bs',0)

#Flight_miles_12mo
ewa['Flight_miles_12mo'].mean() 
ewa['Flight_miles_12mo'].median()
ewa['Flight_miles_12mo'].mode()
ewa['Flight_miles_12mo'].var()
ewa['Flight_miles_12mo'].std()

ewa['Flight_miles_12mo'].skew()
ewa['Flight_miles_12mo'].kurt()
#Skewness = Positive, right skewed
#Kurtosis = Thin and peaked curve
plt.boxplot(ewa['Flight_miles_12mo'],1,'bs',0)

#Days_since_enroll
ewa['Days_since_enroll'].mean()
ewa['Days_since_enroll'].median()
ewa['Days_since_enroll'].mode()
ewa['Days_since_enroll'].var()
ewa['Days_since_enroll'].std()

ewa['Days_since_enroll'].skew()
ewa['Days_since_enroll'].kurt()
#Skewness = weak right skewed
#Kurtosis = flat curve
plt.boxplot(ewa['Days_since_enroll'],1,'bs',0)


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:,1:]) 

# Top 10 rows
df_norm.head(10) 

# Scree plot or elbow curve to find 'k'
distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_norm)
    distortions.append(kmeanModel.inertia_)

#Plotting elbow curve
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-');
plt.xlabel('k');
plt.ylabel('Distortion');
plt.title('The Elbow Method showing the optimal k');
plt.show()

# From elbow curve, it is noticable the curve has a bend at k=4
# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(df_norm)

# Assigning labels of clusters to each row
model.labels_ 
# Converting numpy array into pandas series object 
clusters=pd.Series(model.labels_)  # Converting numpy array into pandas series object 
ewa = ewa.drop('ID#', axis=1) # Removing charater column
# Creating a new column and assigning it to a new column
ewa['clust']= clusters 
ewa.head()
# Putting 'clusters' in the first column
ewa = ewa.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
ewa.head()
# Finding the mean of each cluster w.r.t all variables
ewa_mean = ewa.iloc[:,1:11].groupby(ewa.clust).mean()

#Hierarchical clustering
import pandas as pd
import matplotlib.pylab as plt 
ewa = pd.read_excel("EastWestAirlines.xlsx", sheet_name='data')

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:,1:]) 

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # For creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")

#Plotting of dendrogram
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from sklearn.cluster import	AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

# Creating a new column
cluster_labels=pd.Series(h_complete.labels_)

# Creating a  new column and assigning it to a new column 
ewa['clust']=cluster_labels
ewa = ewa.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
ewa.head()
# Getting aggregate of each cluster
ewa_mean = ewa.iloc[:,1:].groupby(ewa.clust).median()

# Observation : Kmeans clustering model was built on 4 clusters and hierarchical clustering formed 5 clusters