#Importing packages
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

# Kmeans on Crime Data set 
cd = pd.read_csv("crime_data.csv")
colnames = cd.columns
print(colnames)

#EDA
cd.describe()
cd.isnull().sum() # No N.A values

#Business moments and visualizations
#Murder
cd['Murder'].mean() 
cd['Murder'].median()
cd['Murder'].mode()
cd['Murder'].var()
cd['Murder'].std()

cd['Murder'].skew()
cd['Murder'].kurt()
#skewness = moderately right skewed
#kurtosis = flatness in curve
plt.boxplot(cd['Murder'],1,'rs',0)

#Assault
cd['Assault'].mean() 
cd['Assault'].median()
cd['Assault'].mode()
cd['Assault'].var()
cd['Assault'].std()

cd['Assault'].skew()
cd['Assault'].kurt()
#skewness = moderately right skewed
#kurtosis = flatness in curve
plt.boxplot(cd['Assault'],1,'rs',0)

#UrbanPop
cd['UrbanPop'].mean() 
cd['UrbanPop'].median()
cd['UrbanPop'].mode()
cd['UrbanPop'].var()
cd['UrbanPop'].std()

cd['UrbanPop'].skew()
cd['UrbanPop'].kurt()
#skewness = moderately left skewed
#kurtosis = flatness in curve
plt.boxplot(cd['UrbanPop'],1,'rs',0)

#Rape
cd['Rape'].mean()
cd['Rape'].median()
cd['Rape'].mode()
cd['Rape'].var()
cd['Rape'].std()

cd['Rape'].skew()
cd['Rape'].kurt()
#skewness = right skewed
#kurtosis = moderately peaked
plt.boxplot(cd['Rape'],1,'rs',0)


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(cd.iloc[:,1:]) 

# Top 10 rows
df_norm.head(10)  

# Scree plot or elbow curve
distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_norm)
    distortions.append(kmeanModel.inertia_)

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
clusters=pd.Series(model.labels_)  # Converting numpy array into pandas series object 
cd = cd.drop('Unnamed: 0', axis=1) # Removing charater column

# Creating a new column and assigning it to a new column
cd['clust']= clusters 
cd.head()

# Putting 'clusters' in the first column
cd = cd.iloc[:,[4,0,1,2,3]]
cd_mean = cd.iloc[:,1:].groupby(cd.clust).mean()


#Hierarchical clustering#
import pandas as pd
import matplotlib.pylab as plt 
cd = pd.read_csv("crime_data.csv")

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(cd.iloc[:,1:]) 

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # For creating dendrogram 

type(df_norm)

#Plotting of dendrogram
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from sklearn.cluster import	AgglomerativeClustering 
h_complete= AgglomerativeClustering(n_clusters=3, linkage='complete',affinity = "euclidean").fit(df_norm) 

# Creating a new column
cluster_labels=pd.Series(h_complete.labels_)

# Creating a new column and assigning it to a new column
cd['clust']=cluster_labels 
cd = cd.iloc[:,1:]
cd = cd.iloc[:,[4,0,1,2,3]]
cd.head()
# Getting aggregate mean of each cluster
cd_mean = cd.iloc[:,1:].groupby(cd.clust).median()

# Observation : Kmeans clustering model was built on 4 clusters and hierarchical clustering formed 3 clusters