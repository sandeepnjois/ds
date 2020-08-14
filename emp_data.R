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


plot(x,y, xlab = "Salary hike", ylab = "Churn out rate", las=1) # Scatter plot, shows negative linearity and also labelling x,y
cor(x,y)
# r-value = 0.9117 shows -ve strong correlation

# Model building
model <- lm(y ~ x) #Linear regression, on y w.r.t x
model

summary(model)   # summary of all attributes
# p-value is < 0.05
# R-squared value = 0.8312 showing the strength of thw prediction model

#Intervals
confint(model, level = 0.95)  # Confidence intervals at 5% significane level as per industry standards
predict(model, interval = "predict") # Prediction intervals with fit values for future responses    

