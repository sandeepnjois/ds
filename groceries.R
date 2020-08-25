# Groceries

# Loading "arules" and "arulesViz" for transaction dataset
library(arules)
library(arulesViz)


# Importing file
groceries <- read_csv(file.choose())
View(groceries)
class(groceries)

# Converting to transactions data
groceries <- as(groceries, "transactions")
class(groceries)

# First 100 transactions 
inspect(groceries[1:100])
itemFrequencyPlot(groceries)

#1
# Performing apriori algorithm 
rules <- apriori(groceries, parameter = list(support=0.002, confidence=0.5))
rules # Obtained a set of 22 rules 
inspect(rules[1:10])
inspect(head(sort(rules, by="lift")))

# Plot of rules, scatter vs confidence also w.r.t lift
plot(rules)
head(quality(rules))

#2 With different support and confidence 

rules1 <- apriori(groceries, parameter = list(support= 0.004, confidence = 0.7))
rules1
# Only set of 2 rules
plot(rules1)


#3

rules2 <- apriori(groceries, parameter = list(support=0.001, confidence=0.6, minlen=3))
rules2
# Set of 30 rules
plot(rules2, jitter=0)


#4 

rules3 <- apriori(groceries, parameter = list(support=0.0005, confidence=0.5, minlen=4))
rules3
# Set of 42 rules 
plot(rules3, jitter=0)
inspect(rules3[1:10])


