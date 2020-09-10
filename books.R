# Books

#Importing necessary packages 
library(recommenderlab)
library(Matrix)
library(caTools)

#Importing data set
books <- read.csv(file.choose())
View(books)
books <- books[-1] # Removing first column 
View(books)

#Plotting histogram 
hist(books$Book.Rating) #( Highest rating from 6.5 - 8)


#Converting into 'realRatingMatric' datatype 
books_matrix <- as(books, "realRatingMatrix")


#Building model based on popularity
books_recomm_model <- Recommender(books_matrix, method="POPULAR")


#Recommendation of 10 different movies for 5 users
recommended_items <- predict(books_recomm_model,books_matrix[612:616], n=10)
as(recommended_items, "list")



