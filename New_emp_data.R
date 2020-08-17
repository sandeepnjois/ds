# emp_data

library(readr)
ed <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 4 - Simple Linear Regression/emp_data.csv")
View(ed)
attach(ed)

# Assigning two vairables into x and y

x <- ed$Salary_hike
x

y <- ed$Churn_out_rate
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
skewness(x) # Right skewed 
skewness(y) # positive skewed, slightly right skewed

# 4th B.M
kurtosis(x) # Distribution is flat
kurtosis(y) # Distribution is flat

#Boxplots
boxplot(x,las=1, main="Salary Hike")
boxplot(y, las=1, main="Churnout rate")
# Zero outliers




install.packages("ggcorrplot")
library(ggcorrplot)
corr <- round(cor(ed))
ggcorrplot(ed) # Correlation plot





plot(x,y, xlab = "Salary hike", ylab = "Churn out rate", las=1) # Scatter plot, shows negative linearity and also labelling x,y

cor(x,y)
# r-value = 0.9117 shows -ve strong correlation

# Model building
model <- lm(y ~ x) #Linear regression, on y w.r.t x
model

summary(model)   # summary of all attributes



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


# Best fit model with p-values < 0.05 and R squared value of 0.8312
model
summary(model)
#Intervals
confint(model, level = 0.95)  # Confidence intervals at 5% significance level as per industry standards
predict(model, interval = "predict") # Prediction intervals with fit values for future responses    

