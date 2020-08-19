# Customer order form

library(readr)
cof <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 3 - Hypothesis Testing/CustomerOrderForm.csv")
View(cof)

# Stacking data
stacked_data <- stack(cof)
View(stacked_data)
attach(stacked_data)
table(values,ind)



# Chi square test

# H0: Null Hypothesis - The proportions of all defectives are equal
# Ha: Alternative Hypothesis - The proportion of at least one of the centers is unequal
chisq.test(table(values,ind))
# p-value = p-value = 0.2771 > 0.05
# Accept null hypothesis
# All proportions are equal