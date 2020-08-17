# Calories_consumed data



library(readr)
cal <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 4 - Simple Linear Regression/calories_consumed.csv")
View(cal)
attach(cal)

# Assigning two vairables into x and y

x <- cal$`Calories Consumed`     
x     
y <- cal$`Weight gained (grams)`
y

# Finding N.A or missing values

is.na(x)
is.na(y)
# There are NO N.A values 


# Calculating Business moments 
# Central tendencies 
mean(x)
mean(y)

median(x)
median(y)

library(NCmisc)
Mode(x)
Mode(y)

#Standard Deviation
sd(x)
sd(y)

#Variance
var(x)
var(y)

install.packages("e1071") # Installing "e1071" package to find 3rd and 4th Business Moments
library(e1071)

# 3rd B.M 
skewness(x) # right skewed 
skewness(y) # positive skewed, right skewed

# 4th B.M
kurtosis(x) # Distribution is flat
kurtosis(y) # Distribution is slightly flat


#Boxplots
boxplot(x,las=1, main="Calories Consumed")
boxplot(y, las=1, main="Weight gained")
# Zero outliers



# Scatter plots
plot(x,y) # Scatter plot, shows positive linearity 
cor(x,y) # r value is > than 0.85 showing strong correlation
install.packages("ggcorrplot")
library(ggcorrplot)
corr <- round(cor(cal))
ggcorrplot(cal) #correlation plot



# Regressing y with respect to x
model <- lm(y ~ x)
model
summary(model) # summary of all attributes

# Applying transformation in order to obtained a better model
#1
x1 <- log(x)
model1 <- lm(y ~x1)
summary(model1)

#2
x2 <- x^2
model2 <- lm(y ~ x2)
summary(model2)

#3
x3 <- 1/x
model3 <- lm(y ~ x3)
summary(model3)

#4
x4 <- 1/(x^2)
model4 <- lm(y ~ x4)
summary(model4)


#5
y1 <- log(y)
model5 <- lm(y1 ~x)
summary(model5)


# Best fit model with p-values < 0.05 and R squared value of 0.89
model
summary(model) 


# R-squared value = 0.8968, showing the prediction model is good. 

#Intervals
confint(model, level = 0.95) # Confidence intervals 
predict(model, interval = "predict") # Prediction intervals
 