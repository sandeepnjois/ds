#Wine - PCA

library(readr)
wine <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 8 - PCA/wine.csv")
View(wine)
attach(wine)

wine1 <- wine[-1] # Removing column "Type"
View(wine1)
attach(wine1)

# Correlation 
cor(wine1)

# Performing principal component analysis
pcaobj <- princomp(wine1, cor = TRUE, scores = TRUE, covmat = NULL)
summary(pcaobj)

# Plots 
plot(pcaobj,las=1) # Bar charts of different components vs Variances
biplot(pcaobj) # Biplot showing variances of all variables on first 2 components 
pcaobj$scores

# Selecting first 3 components 
pcaobj$scores[,1:3]

# Adding first 3 components columns to the main data, wine
wine <- cbind(wine,pcaobj$scores[,1:3])
View(wine)


# [1] Hierarchial clustering 
clust_data <- wine[,15:17]
View(clust_data)

# Standardizing the clust_data which contains Principal components
norm_clust <- scale(clust_data)
View(norm_clust)

# Distance matrix
distance <- dist(norm_clust)
distance

# Performing hierarchial clustering 
fit <- hclust(distance, method = "complete")
plot(fit, hang = -1)

# Cut tree into 3 clusters
groups <- cutree(fit,3)
groups

rect.hclust(fit, k=3, border = "red") # 3 clusters red bordered 

# Forming a column of groups
membership<-as.matrix(groups)
View(membership)

# Combining groups column with the original dataset
final <- cbind(membership,wine)
View(final)

# [2] K means clustering 

wine1
View(wine1)

# Performing k-means clustering

kmwine <- kmeans(wine1, 3, nstart = 1)
str(kmwine)
kmwine$cluster
kmwine$centers
kmwine$totss
kmwine$withinss
kmwine$tot.withinss
kmwine$betweenss
kmwine$size

library(animation)

kmwine <- kmeans.ani(wine1, 3)
kmwine

# Forming a new column of clusters
membership1 <- as.matrix(kmwine$cluster)
membership1

# Combining membership1 column with the original dataset
final1 <- cbind(membership1, wine)
View(final1)


# By observation, we can infer that k-means clustering is more accurate in forming 3 clusters of similar types 
