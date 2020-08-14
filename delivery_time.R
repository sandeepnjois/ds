#delivery_time

library(readr)
dt <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 4 - Simple Linear Regression/delivery_time.csv")
View(dt)

# Assigning two vairables into x and y

x <- dt$`Sorting Time`
x

y <- dt$`Delivery Time`
y

plot(x,y) # Scatter plot, shows positive linearity
plot(x,y, xlab = "Sorting Time", ylab = "Delivery Time", las=1, main = " Delivery time / Sorting time") #Lavbelling x, y asix and the plot name

cor(x,y) #r-value=0.826 showing moderatly strong correlation


# Model building
model <- lm(y ~ x) # Regressing y with respect to x
model

summary(model)  # summary of all attributes
#Intervals
confint(model, level = 0.95)  # Confidence intervals at 5% significane level as per industry standards
predict(model, interval = "predict")  # Prediction intervals with fit values for future responses    
