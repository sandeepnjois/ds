# ToyotaCorolla dataset

library(readr)
library(dplyr)
library(ggplot2)

ToyotaCorolla <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 5 - Multi Linear Regression/ToyotaCorolla.csv")
tc <- ToyotaCorolla
View(tc)
attach(tc)

tytcr <- select(tc, Price, Age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, Weight)
View(tytcr)

# Assigning variables into onjects

a <- Age_08_04
b <- KM
c <- HP
d <- cc
e <- Doors
f <- Gears
g <- Quarterly_Tax
h <- Weight

# Business Moments

library(NCmisc)  # for Mode function
library(e1071)   # for 3th and 4th B.M's

# Age_08_04

mean(a)
median(a)
Mode(a)

sd(a)
var(a)

is.na(a) # No NA values found

boxplot(a)
plot(a,Price, xlab = "Age_08_04", ylab = "Price", las=1)

cor(a, Price) # Strong negative correlation


# KM

mean(b)
median(b)
Mode(b)

sd(b)
var(b)

is.na(b)   # No NA values found

boxplot(b)
plot(b, Price, xlab = "KM", ylab = "Price", las=1)

cor(b, Price) # Moderate negative correlation

# HP

mean(c)
median(c)
Mode(c)

sd(c)
var(c)

is.na(c)    # No NA values found

boxplot(c)
plot(c, Price, xlab = "HP", ylab = "Price", las=1)
cor(c, Price) # Weak positive correlation 

# cc

mean(d)
median(d)
Mode(d)

sd(d)
var(d)

is.na(d)   # No NA values found

boxplot(d)
plot(d, Price, xlab = "cc", ylab = "Price", las=1)

cor(d, Price) # Low correlation 


# Doors

mean(e)
median(e)
Mode(e)

sd(e)
var(e)

is.na(e)    # No NA values found

boxplot(e)
plot(e, Price, xlab = "Doors", ylab = "Price", las=1)

cor(e, Price) # Low correlation 


# Gears

mean(f)
median(f)
Mode(f)

sd(f)
var(f)

is.na(f)  # No NA values found

boxplot(f)
plot(f,Price, xlab = "Gears", ylab = "Price", las=1)

cor(f, Price) # No correlation

# Quarterly_tax

mean(g)
median(g)
Mode(g)

sd(g)
var(g)

is.na(g)   # No NA values found

boxplot(g)
plot(g, Price, xlab = "Quarterly_Tax", ylab = "Price", las=1)

cor(g, Price) # Low correlation

# Weight

mean(h)
median(h)
Mode(h)

sd(h)
var(h)

is.na(h)   # No NA values found

boxplot(h)
plot(h, Price, xlab = "Weight", ylab = "Price", las=1)

cor(h, Price) # Moderate positive correlation


# Correlation between every variables 
cor(tytcr) # There is no collinearity problem between any two variables, which is a good sign
pairs(tytcr) # Scatter plots of all variable combinations


attach(tc)

# Building regression model

model.tytcr <- lm(Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight)
summary(model.tytcr)
# p-values of 'cc' and 'doors' are more than 0.05

#Finding the influencial data
library(mvinfluence)

influence.measures(model.tytcr)
influenceIndexPlot(model.tytcr) 

# It is clear from the diagnostic plots that Observation no. 81 has influence in building the right prediction model
# p-values of cc and Doors are insignificant 



model.tytcr2 <- lm(Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight, data = tytcr[-c(81),])
summary(model.tytcr2)

# by deleting the row 81, we can notice that the p-values of variables Doors and cc have significantly reduced 
# But p-value of Doors is > 0.05

# Deleting 222nd and 961th observations, as they are observed to have influence from diagnostic plots 

model.tytcr3 <- lm(Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight, data = tytcr[-c(81,222,961),])
summary(model.tytcr3)

#R-squared value has increased
# The difference between R-squared and multiple R-squared values is negligible, even after deleting 3 observations
# Also the overall p-value is < 0.05


#Final model

Finalmodel <- model.tytcr3
Finalmodel

# Residual qqnorm plot

qqnorm(model.tytcr3$residuals) # residuals of the model follow a normal distribution

# Intervals

confint(model.tytcr3, level = 0.95)
predict(model.tytcr3, interval = "predict")




