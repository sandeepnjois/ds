# Salary_data

library(readr)
sal <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 4 - Simple Linear Regression/Salary_data.csv")
View(sal)
attach(sal)

# Assigning two vairables into x and y
x <- sal$YearsExperience
x

y <- sal$Salary
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
skewness(x) # positive skewed, slightly right skewed
skewness(y) # positive skewed, slightly right skewed

# 4th B.M
kurtosis(x) # Distribution is flat
kurtosis(y) # Distribution is flat

#Boxplots
boxplot(x,las=1, main="Years of experience")
boxplot(y, las=1, main="Salary")
# Zero outliers




install.packages("ggcorrplot")
library(ggcorrplot)
corr <- round(cor(sal))
ggcorrplot(sal) # Correlation plot


#Scatter plot
plot(x,y, xlab = "Years of experience", ylab = "Salary", las=1, main = "Salary Hike w.r.t experience") # Scatter plot, shows positive linearity
#Labeling x, y axes and the plot name

cor(x,y) #r-value=0.9782 showing a very strong correlation




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


# Building model
model <- lm(y ~ x) #Linear regression, on y w.r.t x
model
summary(model) # summary of all attributes
# p-value is < 0.05
# R-squared value = 0.957 showing the strength of the prediction model

# Intervals
confint(model, level = 0.95)  # Confidence intervals at 5% significane level as per industry standards
predict(model, interval = "predict")# Prediction intervals with fit values for future responses   


