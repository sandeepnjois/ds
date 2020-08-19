# BuyerRatio

library(readr)
br <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 3 - Hypothesis Testing/BuyerRatio.csv")
View(br)
attach(br)


table(`Observed Values`, East, West, North, South)
br1 <- br[-1] # Removing first column of gender
View(br1)
attach(br1)


# Performing chi square test
# Ha: Null hypothesis : All proportions are equal
# H0: Alternative hypothesis : One of the proportions is unequal
chisq.test(br1)
# p-value = 0.66> 0.05
# Accept null hypothesis 
# Male-female buyer ratios are similar across all regions