# Books

#Loading "arules" and "arulesViz" for transaction dataset
library(readr)
library(arules)
library(arulesViz)

# Importing file
books <- read_csv(file.choose())
View(books)
str(books)
dim(books)
class(books)


# Converting to transactions data
books <- as(books, "transactions")
class(books)

# First 100 transactions 
inspect(books[1:100])


# Performing apriori algorithm
#1
rules1 <- apriori(books, parameter= list(support=0.1, confidence=0.5))
rules1                       # 11253 rules
plot(rules1)

#2
rules2 <- apriori(books, parameter = list(support = 0.01, confidence=0.5, minlen=3))
rules2                      # 11132 rules
plot(rules2)

#3
rules3 <- apriori(books, parameter = list(support = 0.1, confidence=0.5, minlen=4))
rules3                      # 10637 rules
plot(rules3)

#4
rules4 <- apriori(books, parameter = list(support = 0.01, confidence=0.5, minlen=5))
rules4                      # 9317 rules
plot(rules4)

#5
rules5 <- apriori(books, parameter = list(support = 0.01, confidence=0.5, minlen=6))
rules5                      # 7007 rules
plot(rules5)

#6
rules6 <- apriori(books, parameter = list(support = 0.01, confidence=0.5, minlen=7))
rules6                 # 4235 rules
plot(rules6)

#7
rules7 <- apriori(books, parameter = list(support = 0.01, confidence=0.5, minlen=8))
rules7                      # 1925 rules
plot(rules7)

#8
rules8 <- apriori(books, parameter = list(support = 0.01, confidence=0.5, minlen=9))
rules8                    # 605 rules  
plot(rules8)

#9
rules9 <- apriori(books, parameter = list(support = 0.01, confidence=0.5, minlen=10))
rules9                     # 110 rules
plot(rules9)


#10
rules10 <- apriori(books, parameter = list(support = 0.1, confidence=0.5, minlen=10))
rules10                     # 110 rules
plot(rules10)

#11
rules11 <- apriori(books, parameter = list(support = 0.9, confidence=0.9, minlen=10))
rules11                     # 110 rules
plot(rules11)

inspect(head(sort(rules11, by="lift")))

# rules11 gives 110 rules with best support and confidence values. Hence, good inferences may be made from these rules