# LabTAT

library(readr)

LabT <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 3 - Hypothesis Testing/LabTAT.csv")
View(LabT)
attach(LabT)

# Normality test on every variable 
shapiro.test(`Laboratory 1`) # p-value = 0.55> 0.05
shapiro.test(`Laboratory 2`) # p-value = 0.86> 0.05
shapiro.test(`Laboratory 3`) # p-value = 0.42> 0.05
shapiro.test(`Laboratory 4`) # p-value = 0.66> 0.05

# All 4 variables follow a normal distribution. 
# Proceeding with variance test

stacked_data <- stack(LabT)
View(stacked_data)

# Proceeding with variance test
install.packages("car")
library(car)
attach(stacked_data)
leveneTest(values, ind, data= "stacked_data") # p-value> 0.05 # Equal variances

# Proceeding with ANOVA test

# H0 <- Average of all variables are equal
# Ha <- Average of one of the variables is unequal 
Anova_results <- aov(values ~ ind)
Anova_results
summary(Anova_results)
#p-value < 0.05
# Reject null hypothesis
# Any one or more of the TAT average value is unequal to the rest