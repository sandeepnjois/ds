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

plot(x,y, xlab = "Years of experience", ylab = "Salary", las=1, main = "Salary Hike w.r.t experience") # Scatter plot, shows positive linearity
#Lavbelling x, y asix and the plot name

cor(x,y) #r-value=0.9782 showing a very strong correlation

# Building model
model <- lm(y ~ x) #Linear regression, on y w.r.t x
model
summary(model) # summary of all attributes
# p-value is < 0.05
# R-squared value = 0.957 showing the strength of the prediction model

# Intervals
confint(model, level = 0.95)  # Confidence intervals at 5% significane level as per industry standards
predict(model, interval = "predict")# Prediction intervals with fit values for future responses   


