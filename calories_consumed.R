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

plot(x,y) # Scatter plot, shows positive linearity 
cor(x,y) # r value is > than 0.85 showing strong correlation

# Regressing y with respect to x
model <- lm(y ~ x)
model
summary(model) # summary of all attributes

# R-squared value = 0.8968, showing the prediction model is good. 

#Intervals
confint(model, level = 0.95) # Confidence intervals 
predict(model, interval = "predict") # Prediction intervals
 