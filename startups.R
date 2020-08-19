# 50_Startups

library(readr)
startups <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 5 - Multi Linear Regression/50_Startups.csv")
View(startups)
attach(startups)

summary(startups)

a <- startups$`R&D Spend`
a

b <- startups$Administration
b

c <- startups$`Marketing Spend`
c

y <- startups$Profit
y

# Business moments 
library(NCmisc)
library(e1071)

mean(a)
mean(b)
mean(c)

median(a)
median(b)
median(c)

Mode(a)
Mode(b)
Mode(c)


sd(a)
sd(b)
sd(c)

var(a)
var(b)
var(c)

is.na(a) # No N.A values
is.na(b) # No N.A values
is.na(c) # No N.A values


# Boxplot

boxplot(a,y, xlab="R&D Spend", ylab="Profit", las=1)
boxplot(b,y,xlab="Administration", ylab="Profit", las=1)
boxplot(c,y,xlab="Marketing Spend", ylab="Profit", las=1)

# Scatter plot

plot(a,y, xlab = "R&D Spend", ylab = "Profit", las=1)
plot(b,y, xlab = "Administration", ylab = "Profit", las=1)
plot(c,y, xlab = "Marketing Spend", ylab = "Profit", las=1)

# Correlation 

cor(a,y) # Strong positive correlation
cor(b,y) # Weak positive correlation
cor(c,y) # Moderate positive correlation


library(dplyr)

# Removing the 'State' variable
select(startups, -State)
View(startups)
stup <- select(startups, -State)
View(stup)
pairs(stup)
# As per scatter plot, there is collinearity between  (R$D Spend & Marketing Spend)

cor(stup)

install.packages("corpcor")
library(corpcor)
cor2pcor(cor(stup))
# Partial correlation coeffiicient does not show collinearity between any of the variables

# Building linear model

model.startups <- lm(y ~ a+b+c)
model.startups
summary(model.startups)
# p-values of Administration and Marketing speed are insignificant

# Performing transformation on 'Administration' and 'Marketing'

b1 <- log(b)
b2 <- b^2
b3 <- 1/b

c1 <- log(c)
c2 <- c^2
c3 <- 1/c

mod1 <- lm(y ~ a+b1+c)
summary(mod1)

mod2 <- lm(y ~ a+b2+c)
summary(mod2)

mod3 <- lm(y ~ a+b3+c)
summary(mod3)

mod4<- lm(y ~ a+b+c2)
summary(mod4)

mod5 <- lm(y ~ a+b1+c2)
summary(mod5)

mod6 <- lm(y ~ a+b2+c2)
summary(mod6)

# No significane in p-values even after transformations

install.packages("mvinfluence")
library(mvinfluence)
influence.measures(model.startups)
influenceIndexPlot(model.startups)
# 50,49,47,20 are the observations with high influence 

mod7 <- lm(y ~ a+b+c, data = stup[-c(20,47,49,50),])
summary(mod7)


# p-values are still greater than 0.05 after removing 4 influencial observations

vif(model.startups)

mod8 <- lm(y ~ b2+c)
summary(mod8) # p-value for 'Marketing spend' and 'Administration' is < 0.05 with R-squared = 0.61

mod9 <- lm(y ~ a+b)
summary(mod9) # p-value for 'Marketing spend'= 0.28 > 0.05

mod10 <-  lm(y ~ a+c)
summary(mod10) # p-value for 'Marketing spend'= 0.06 >0.05

mod11 <-  lm(y ~ a+c, data = st)
summary(mod11)  # p-value for 'Marketing spend'= 0.06 >0.05


# Final model 
Finalmodel <- mod8
Finalmodel

# Intervals
confint(Finalmodel, level = 0.95)
predict(Finalmodel, interval = "predict")


# R^2 for different models

# Model 1=  0.9506
# Model 2=  0.9508
# Model 3=  0.9505
# Model 4=  0.9496
# Model 5=  0.9494
# Model 6=  0.9497
# Model 7=  0.9507
# Model 8=  0.6099 ( BEST FIT WITH  p-values < 0.05)
# Model 9=  0.9478
# Model 10= 0.9505
# Model 11= 0.9505
# Model.startups = 0.9475