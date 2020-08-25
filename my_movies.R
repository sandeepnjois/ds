# MyMovies

#Loading "arules" and "arulesViz" for transaction dataset

library(readr)
library(arules)
library(arulesViz)

# Importing file

movies <- read_csv(file.choose())
View(movies)
dim(movies)
str(movies)
class(movies)

# Removing first 5 columns of movie names
library(dplyr)
mymovies <- select(movies, -c(V1,V2,V3,V4,V5))
View(mymovies)
sum(is.na(mymovies)) # No N.A values found

# Converting to transactions data

mymovies <- as(mymovies, "transactions")
class(mymovies)
inspect(mymovies[1:5])# First 5 transactions 

# Assigning mymovies to "m"
m <- mymovies


# Performing apriori algorithm

#1
r1 <- apriori(m, parameter = list(support=0.2, confidence=0.5))
r1   # 5120 rules
plot(r1)

#2
r2 <- apriori(m, parameter = list(support=0.2, confidence=0.5, minlen=3))
r2      # 5020 rules
plot(r2)

#3
r3 <- apriori(m, parameter = list(support=0.2, confidence=0.5, minlen=4))
r3     # 4660 rules
plot(r3)

#4
r4 <- apriori(m, parameter = list(support=0.2, confidence=0.5, minlen=8))
r4     # 460 rules
plot(r4)

#5
r5 <- apriori(m, parameter = list(support=0.2, confidence=0.5, minlen=9))
r5    # 100 rules
plot(r5)

#6
r6 <- apriori(m, parameter = list(support=0.99, confidence=0.9, minlen=9))
r6     # 100 rules
plot(r6)

inspect(head(sort(r6, by="lift")))
# r6 gives 100 rules with best support and confidence values. Hence, good inferences may be made from these rules